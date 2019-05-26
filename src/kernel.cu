/*
* Name: cuda_pip.cuh
* Author  : Evgenii Vasilev
* Created : 05.04.2016
* Description: Main function in mesher program
* Version: 1.0
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_pip.cuh"
#include "cuda_meshgen.cuh"
#include "helper_math.cuh"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

#include "grainmesh.h"
#include "iotetgen.h"
#include "mystl.h"
#include "mymesh.h"
#include "meshcut.h"
#include "meshsmooth.h"
#include <ctime>

#define VarToStr(v) #v
using grainSts = GRAIN_STATUS;

void convertMeshToArrays(grain::GrainMesh* mesh, float3* &vertices, std::vector<float> * tr, float3* &triangles);

void printHelp() {
    std::cout << "You can use following flags: " << std::endl;
    std::cout << " -nX (nY, nZ) to specify mesh size (optional) " << std::endl;
    std::cout << " -offX (offY, offZ) to specify mesh offset (optional)" << std::endl;
    std::cout << " -edgeLen to specify edge length (optional, default 0.2) " << std::endl;
    std::cout << " -generateMeshGPU to generate mesh on GPU (optional, default true) " << std::endl;
    std::cout << " -saveMeshAfterGenerate to save generated mesh (optional, default true) " << std::endl;
    std::cout << " -loadMeshBeforeMark load generated mesh for mark (optional, default false) " << std::endl;
    std::cout << " -markMeshGPU to mark mesh on GPU (optional, default true) " << std::endl;
    std::cout << " -saveMeshAfterMark to save marked mesh (optional, default true) " << std::endl;
    std::cout << " -loadMeshBeforeCut to load marked mesh before Cut (optional, default false) " << std::endl;
    std::cout << " -cutMesh to cut mesh(optional, default true) " << std::endl;
    std::cout << " -saveMeshAfterCut to save mesh generation (optional, default true) " << std::endl;
    std::cout << " -loadMeshBeforeSmooth to load cut mesh before smoth (optional, default false) " << std::endl;
    std::cout << " -smoothMesh to start mesh smooth (optional, default true) " << std::endl;
    std::cout << " -saveMeshAfterSmooth to save marked mesh afterSmooth (optional, default true) " << std::endl;
    std::cout << " -i path to input STL file (optional, default hardcode) " << std::endl;
    std::cout << " -o path to results and tmp files (optional, default hardcode) " << std::endl;
}

bool Validate(const int &lhs, const int &rhs) {
    return lhs >= rhs;
}

grainSts ValidateAndAddtoMap(char * input_args[], const int& count_arg, int &i, std::map<std::string, std::string>& args) {
    if (Validate(i + 1, count_arg)) {
        return GRAIN_ERR_WRONG_PARAMETER;
    }
    else {
        args[input_args[i]] = input_args[++i];
        return GRAIN_ERR_NONE;
    }
}

void AddSimpleFlag(char * input_args[], const int& count_arg, int &i, std::map<std::string, std::string>& args) {
    args[input_args[i]] = "1";
}

grainSts parseInputFlags(char* input_args[], const int& count_arg, std::map<std::string, std::string>& args) {
    if (count_arg == 1) {
        std::cout << "Run without parameters, set to defaults..." << std::endl;
        return GRAIN_ERR_NONE;
    }

    for (int i = 0; i < count_arg; i++) {

        if (input_args[i] == "-h") {
            printHelp();
            return GRAIN_ERR_NONE;
        }
        else if (input_args[i] == "-nX") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else if (input_args[i] == "-nY") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else if (input_args[i] == "-nZ") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else if (input_args[i] == "-offX") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else if (input_args[i] == "-offY") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else if (input_args[i] == "-offZ") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else if (input_args[i] == "-generateMeshGPU") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-saveMeshAfterGenerate") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-loadMeshBeforeMark") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-markMeshGPU") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-saveMeshAfterMark") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-loadMeshBeforeCut") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-cutMesh") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-saveMeshAfterCut") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-loadMeshBeforeSmooth") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-smoothMesh") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-saveMeshAfterSmooth") {
            AddSimpleFlag(input_args, count_arg, i, args);
        }
        else if (input_args[i] == "-i") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else if (input_args[i] == "-o") {
            if (ValidateAndAddtoMap(input_args, count_arg, i, args) == GRAIN_ERR_WRONG_PARAMETER)
                return GRAIN_ERR_WRONG_PARAMETER;
        }
        else {
            return GRAIN_ERR_WRONG_PARAMETER;
        }
    }
    return GRAIN_ERR_NONE;
}

std::string getDefault(std::string param_name) {
    std::map<std::string, std::string> default_flags = {
        {"nX", "75"}, {"nY", "75"}, {"nZ", "75"},
        { "offX", "0.0" }, { "offX", "0.0" }, { "offZ", "0.0" }, {"egdeLen", "3"},
        { "generateMeshGPU"," 1" },
        { "saveMeshAfterGenerate"," 1" },
        { "loadMeshBeforeMark"," 0" },
        { "markMeshGPU"," 0" },
        { "saveMeshAfterMark"," 0" },
        { "loadMeshBeforeCut"," 0" },
        { "cutMesh"," 0" },
        { "saveMeshAfterCut"," 0" },
        { "loadMeshBeforeSmooth"," 0" },
        { "smoothMesh"," 0" },
        { "saveMeshAfterSmooth"," 0" },
        { "i"," D:\\study\\Graphics\\Parallel-Mesh-Generator\\stl\\00_heart_shell.stl" },
        { "o"," D:\\study\\Graphics\\node\\" }
    };
    return default_flags[param_name];
}

template <class T>
T convert_to_template(const std::string &str) {
    std::istringstream ss(str);
    T num;
    ss >> num;
    return num;
}

template <class T>
void SetParams(std::map<std::string, std::string> flags, T& param, std::string param_name) {
    param = convert_to_template<T>((flags["-" + param_name] == "") ? getDefault(param_name) : flags["-" + param_name]);
}

int main(int argc, char* argv[]) {
    ///
    /// Change parameters here ///
    ///
    std::cout << "Parsing parameters" << std::endl;
    std::map<std::string, std::string> flags;
    // Filename of heart shell // 
    std::string fileHeartStl;
    //fileHeartStl = "E:/Data/STL/CyberheartModel/00_heart_shell.stl";
    // Folder to store mesh results //
    std::string folderpath;
    grainSts sts;
    sts = parseInputFlags(argv, argc, flags);

    // Set mesh parameters //
    int nX, nY, nZ;
    float offX, offY, offZ;
    float edgeLen;

    bool generateMeshGPU;
    bool saveMeshAfterGenerate;
    bool loadMeshBeforeMark;
    bool markMeshGPU;
    bool saveMeshAfterMark;
    bool loadMeshBeforeCut;
    bool cutMesh;
    bool saveMeshAfterCut;
    bool loadMeshBeforeSmooth;
    bool smoothMesh;
    bool saveMeshAfterSmooth;

    if (sts == GRAIN_ERR_NONE) {
        SetParams(flags, nX, VarToStr(nX));
        SetParams(flags, nY, VarToStr(nY));
        SetParams(flags, nZ, VarToStr(nZ));
        SetParams(flags, offX, VarToStr(offX));
        SetParams(flags, offY, VarToStr(offY));
        SetParams(flags, offZ, VarToStr(offZ));
        SetParams(flags, edgeLen, VarToStr(edgeLen));
        SetParams(flags, generateMeshGPU, VarToStr(generateMeshGPU));
        SetParams(flags, saveMeshAfterGenerate, VarToStr(saveMeshAfterGenerate));
        SetParams(flags, loadMeshBeforeMark, VarToStr(loadMeshBeforeMark));
        SetParams(flags, markMeshGPU, VarToStr(markMeshGPU));
        SetParams(flags, saveMeshAfterMark, VarToStr(saveMeshAfterMark));
        SetParams(flags, loadMeshBeforeCut, VarToStr(loadMeshBeforeCut));
        SetParams(flags, cutMesh, VarToStr(cutMesh));
        SetParams(flags, saveMeshAfterCut, VarToStr(saveMeshAfterCut));
        SetParams(flags, loadMeshBeforeSmooth, VarToStr(loadMeshBeforeSmooth));
        SetParams(flags, smoothMesh, VarToStr(smoothMesh));
        SetParams(flags, saveMeshAfterSmooth, VarToStr(saveMeshAfterSmooth));
        SetParams(flags, fileHeartStl, VarToStr(i));
        SetParams(flags, folderpath, VarToStr(o));
    }
    else if (sts == GRAIN_ERR_WRONG_PARAMETER) {
        std::cout << "Wrong parameters" << std::endl;
        printHelp();
        return 0;
    }


    ///
    /// Change parameters here  ///
    ///


    int pCount = nX * nY * nZ;
    int tCount = (nX - 1)*(nY - 1)*(nZ - 1) * 6;
    MyMesh mymesh;
    MySTL stl; stl.readSTL(fileHeartStl);


    float3 *dev_points = 0;
    int4 *dev_tetra = 0;

    clock_t  timeMeshGenStart = 0, timeMeshGenEnd = 0,
        timeMeshMarkStart = 0, timeMeshMarkEnd = 0,
        timeMeshCutStart = 0, timeMeshCutEnd = 0,
        timeMeshSmoothStart = 0, timeMeshSmoothEnd = 0;

    // Generate mesh // 
    if (generateMeshGPU) {
        std::cout << "Start meshgen..." << std::endl;
        // Generate mesh with CUDA //
        timeMeshGenStart = clock();
        cudaError_t cudaStatus = genMeshWithCuda(dev_points, dev_tetra,
            nX, nY, nZ, offX, offY, offZ, edgeLen);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "genMeshWithCuda failed!");
            return 1;
        }
        timeMeshGenEnd = clock();
        std::cout << "Meshgen finished successful!" << std::endl;
    }

    // Save mesh after generate //
    if (saveMeshAfterGenerate) {
        std::cout << "Start saving generated mesh..." << std::endl;
        // Copy Mesh from GPU and save in file //
        float3 *points = new float3[pCount];
        int4 *tetra = new int4[tCount];
        copyMeshFromGPU(points, dev_points, pCount,
            tetra, dev_tetra, tCount);

        // Save mesh //
        MyMesh mymesh;
        mymesh.mPoints = points;
        mymesh.mPointsCount = pCount;
        short* pLabels = new short[pCount];
        for (int i = 0; i < pCount; i++)
            pLabels[i] = 0;
        mymesh.mPointLabels = pLabels;

        mymesh.mTetra = tetra;
        mymesh.mTetraCount = tCount;
        short* tLabels = new short[tCount];
        for (int i = 0; i < tCount; i++)
            tLabels[i] = 0;
        mymesh.mTetraLabels = tLabels;

        grain::saveNodeFile(folderpath + "meshGenerated.node", &mymesh);
        grain::saveEleFile(folderpath + "meshGenerated.ele", &mymesh);
        std::cout << "Mesh successful saved to " <<
            folderpath << "meshGenerated.node" << std::endl;
    }

    if (loadMeshBeforeMark) {
        grain::readNodeFile(folderpath + "torscoloredremoved.node", &mymesh);
        //grain::readEleFile(folderpath + "torscoloredremoved.ele", &mymesh);

        pCount = mymesh.mPointsCount;
        //tCount = mymesh.mTetraCount;

        cudaError_t cudaStatus;
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? \n");
        }
        // Allocate GPU buffers for points
        cudaStatus = cudaMalloc((void**)&dev_points, pCount * sizeof(float3));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! \n");
        }
        float3* p = mymesh.mPoints;
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_points, p,
            pCount * sizeof(float3), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy 1  failed! \n");
        }

        // Copy input vectors from host memory to GPU buffers.
        //cudaStatus = cudaMemcpy(dev_tetra, mymesh.mTetra,
        //	tCount * sizeof(int4), cudaMemcpyHostToDevice);
        //if (cudaStatus != cudaSuccess) {
        //	fprintf(stderr, "cudaMemcpy 1  failed! \n");
        //}

    }

    // Mark labels with CUDA // 
    if (markMeshGPU) {
        float timeWithCopy, timeWithoutCopy;
        float3* mystl = new float3[stl.trigs.size() / 3];
        for (uint i = 0; i < stl.trigs.size() / 3; i++) {
            mystl[i].x = stl.trigs[3 * i + 0];
            mystl[i].y = stl.trigs[3 * i + 1];
            mystl[i].z = stl.trigs[3 * i + 2];
        }
        bool * result = new bool[pCount];
        timeMeshMarkStart = clock();
        // Mark mesh with CUDA //
        cudaError_t cudaStatus = calcIntersectionCuda2(result,
            dev_points, pCount,
            mystl, stl.trigs.size() / 9,
            timeWithCopy, timeWithoutCopy);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "calcIntersectionCuda failed! \n");
            return 1;
        }
        timeMeshMarkEnd = clock();

        // Generate mesh with labels //
        short* resvec = new short[pCount];
        for (int i = 0; i < pCount; i++) {
            int val = 999;
            if (result[i] == true)
                val = 0;
            resvec[i] = val;
        }

        /// Copy Mesh from GPU and save in file //

        float3 *points = new float3[pCount];
        int4 *tetra = new int4[tCount];
        copyMeshFromGPU(points, dev_points, pCount,
            tetra, dev_tetra, tCount);

        mymesh.mPoints = points;
        mymesh.mPointsCount = pCount;
        mymesh.mPointLabels = resvec;

        mymesh.mTetra = tetra;
        mymesh.mTetraCount = tCount;
        short* tLabels = new short[tCount];
        for (int i = 0; i < tCount; i++)
            tLabels[i] = 0;
        mymesh.mTetraLabels = tLabels;
    }

    // Save mesh after mark // 
    if (saveMeshAfterMark) {
        grain::saveNodeFile(folderpath + "meshMarked.node", &mymesh);
        grain::saveEleFile(folderpath + "meshMarked.ele", &mymesh);
    }

    // Load mesh before cut // 
    if (loadMeshBeforeCut) {
        grain::readNodeFile(folderpath + "meshMarked.node", &mymesh);
        grain::readEleFile(folderpath + "meshMarked.ele", &mymesh);
    }

    // Mesh cutting //
    if (cutMesh) {
        timeMeshCutStart = clock();
        MeshCut cut;
        cut.cutMeshMarkedVertices(&mymesh);
        timeMeshCutEnd = clock();
    }

    // Save mesh before cut //
    if (saveMeshAfterCut) {
        grain::saveNodeFile(folderpath + "afterCut.node", &mymesh);
        grain::saveEleFile(folderpath + "afterCut.ele", &mymesh);
    }

    // Load mesh before smooth //
    if (loadMeshBeforeSmooth) {
        grain::readNodeFile(folderpath + "afterCut.node", &mymesh);
        grain::readEleFile(folderpath + "afterCut.ele", &mymesh);
    }

    // Smoothing //
    if (smoothMesh) {
        timeMeshSmoothStart = clock();
        MeshSmooth smooth;
        smooth.edgelen = edgeLen;
        smooth.smoothMesh(&mymesh, &stl);
        timeMeshSmoothEnd = clock();
    }

    // Save smooth after smooth //
    if (saveMeshAfterSmooth) {
        grain::saveNodeFile(folderpath + "afterSmooth.node", &mymesh);
        grain::saveEleFile(folderpath + "afterSmooth.ele", &mymesh);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed! \n");
        return 1;
    }*/

    std::ofstream fout(folderpath + "result.txt");
    fout << " Time mesh generate " << static_cast<float>(timeMeshGenEnd - timeMeshGenStart) / CLK_TCK << "\n"
        << " Time mesh mark " << static_cast<float>(timeMeshMarkEnd - timeMeshMarkStart) / CLK_TCK << "\n"
        << " Time mesh cut " << static_cast<float>(timeMeshCutEnd - timeMeshCutStart) / CLK_TCK << "\n"
        << " Time mesh smooth " << static_cast<float>(timeMeshSmoothEnd - timeMeshSmoothStart) / CLK_TCK << "\n";
    fout.close();

    std::cout << " Time mesh generate " << static_cast<float>(timeMeshGenEnd - timeMeshGenStart) / CLK_TCK << "\n"
        << " Time mesh mark " << static_cast<float>(timeMeshMarkEnd - timeMeshMarkStart) / CLK_TCK << "\n"
        << " Time mesh cut " << static_cast<float>(timeMeshCutEnd - timeMeshCutStart) / CLK_TCK << "\n"
        << " Time mesh smooth " << static_cast<float>(timeMeshSmoothEnd - timeMeshSmoothStart) / CLK_TCK << "\n";

    return 0;
}

void convertMeshToArrays(grain::GrainMesh* mesh, float3* &vertices, std::vector<float> * tr, float3* &triangles) {
    std::vector<vec3d>* vert = mesh->getVertices();
    vertices = new float3[mesh->getVerticesCount()];
    for (int i = 0; i < mesh->getVerticesCount(); i++) {
        vertices[i].x = vert->at(i).x;
        vertices[i].y = vert->at(i).y;
        vertices[i].z = vert->at(i).z;
    }
    triangles = new float3[tr->size() / 3];
    for (size_t i = 0; i < tr->size() / 3; i++) {
        triangles[i].x = tr->at(i * 3 + 0);
        triangles[i].y = tr->at(i * 3 + 1);
        triangles[i].z = tr->at(i * 3 + 2);
    }
}

