#include "meshsmooth.h"
#include <fstream>
#include <algorithm>

float dist(float3 & a, float3 & b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) +
        (a.y - b.y) * (a.y - b.y) +
        (a.z - b.z) * (a.z - b.z));
}

void MeshSmooth::smoothMesh(MyMesh * mesh, MySTL * stl)
{
    fillAdjunctionsList(mesh, adjenctionList);

    //points
    float3* p = mesh->mPoints;
    short * pLabels = mesh->mPointLabels;
    int pCount = mesh->mPointsCount;

    //tetras
    int4* v = mesh->mTetra;
    int vCount = mesh->mTetraCount;


    std::vector<int> endpoints = std::vector<int>();
    //����������������� ������ "-1", ����� ����� ���� ������ ������� "��������" ��������.
    endpoints.assign(pCount, -1);

    //������ ��� ������ ������ ����� �������� ����� � �� ����� ������ ����� � ������

    for (int i = 0; i < pCount; i++)
    {
        if (pLabels[i] > 100)
        {
            //endpoints.push_back(i);
            endpoints.at(i) = i;
        }
    }
    //erase ������� ��� �������� ������� � begin �� end
    //remove ���������� ������� �������� ��������� � ����� �������
    endpoints.erase(
        std::remove(endpoints.begin(), endpoints.end(), -1),
        endpoints.end());
    //� ����� �� ������� ��� �� ������, ��� ��� �� �����, ��
    //������ ����������� ������������� ���, �.�. ����� ���������� ������ �� ������������.

    // ��������� ����������� ����� ������ ����  
    float3* newEndPoints = new float3[endpoints.size()];


    for (int i = 0; i < endpoints.size(); i++)
    {
        float3 pp = make_float3(0.0f, 0.0f, 0.0f);
        int pNumber = endpoints.at(i);
        std::set<int>* neighbours = adjenctionList->at(pNumber);
        for (std::set<int>::iterator it = neighbours->begin(); it != neighbours->end(); ++it)
        {
            int curPointNumber = *it;
            pp += p[curPointNumber];
        }
        pp /= neighbours->size();
        newEndPoints[i] = pp;

    }

    std::vector<int> forDelete = std::vector<int>();

    // ������� ���������� ��������� ����� //

//����� ������ ����������� push_back forDelete
    // ������� ��� �� ��� ����, �.�. ����� ������� ����� ��� forDelete � ������� -1
    forDelete.assign(endpoints.size(), -1);

    for (int i = 0; i < endpoints.size(); i++)
    {
        int movePointNumber = endpoints[i];
        float3 oldPoint = p[movePointNumber];
        float tNearest = 100.0f;

        for (int j = 0; j < stl->trigs.size(); j += 9)
        {
            float t = hasIntersection(
                oldPoint,
                make_float3(newEndPoints[i].x, newEndPoints[i].y, newEndPoints[i].z),
                make_float3(stl->trigs[j + 0], stl->trigs[j + 1], stl->trigs[j + 2]),
                make_float3(stl->trigs[j + 3], stl->trigs[j + 4], stl->trigs[j + 5]),
                make_float3(stl->trigs[j + 6], stl->trigs[j + 7], stl->trigs[j + 8]));
            if ((t > 0.0f) && (t < tNearest))
                tNearest = t;
        }

        float3 newPoint = make_float3(newEndPoints[i].x, newEndPoints[i].y, newEndPoints[i].z) - oldPoint;
        if (length(newPoint) > 0.001f)
            newPoint = normalize(newPoint);

        if (tNearest > 0.99f) tNearest = 0.0f;

        newPoint = newPoint * tNearest + oldPoint;

        // ���� ����� ������� ������ �� ����, �� �� ����� �������, //
        // � �� ������ ������� ���� ��������� //
        if (length(newPoint - p[movePointNumber]) > edgelen * 0.5)
        {
            //forDelete.push_back(movePointNumber);
            //������ ����� �������� �� ����� i, ��� ����� ����� ��� ��������������, ��� ���
            //�����
            forDelete.at(i) = movePointNumber;

            // ������� ���������� ������ //
            std::set<int>* neighbours = adjenctionList->at(movePointNumber);
            int nearestNumber = *neighbours->begin();
            float nearestDist = INFINITY;
            for (std::set<int>::iterator it = neighbours->begin(); it != neighbours->end(); ++it)
            {
                int curPointNumber = *it;
                float dist = length(newPoint - p[curPointNumber]);
                if (dist < nearestDist)
                {
                    nearestNumber = curPointNumber;
                    nearestDist = dist;
                }
            }
            // ����������� ��� //
            //if(length(newPoint - p[nearestNumber]) < edgelen * 0.5)
            p[nearestNumber] = newPoint;

        }
        p[movePointNumber] = newPoint;
    }

    //������ ������ �� for delete;
    forDelete.erase(
        std::remove(forDelete.begin(), forDelete.end(), -1),
        forDelete.end());
    //������ ���� ���� �� ������� �� ��������
    // Generate relations list //
    int newpCount = 0;
    std::vector<int2>* pairs = generateOldNewPairs(mesh, &forDelete, newpCount);

    // Delete far points //
    MeshCut cut;
    cut.deleteNonRelationPoints(mesh, pairs, newpCount);
    cut.deleteNonRelationTetra(mesh, pairs);

    ///std::cout << "Bad points after cutting \n";
    ///for (int i = 0; i < mesh->mPointsCount; i++)
    ///{
    ///	if (mesh->mPoints[i].y < -7.f || mesh->mPoints[i].x < -7.f || mesh->mPoints[i].z < -7.f)
    ///	{
    ///		std::cout << i << '_';
    ///	}
    ///}
    ///std::cout << "Number of vertices" << mesh->mPointsCount;


    pairs->clear();
}

void MeshSmooth::fillAdjunctionsList(MyMesh * mesh, std::vector<std::set<int>*>*& adjList)
{
    if (adjList == nullptr)
        adjList = new std::vector<std::set<int>*>();

    //points
    float3* p = mesh->mPoints;
    short * pLabels = mesh->mPointLabels;
    int pCount = mesh->mPointsCount;
    //tetras
    int4* v = mesh->mTetra;
    int vCount = mesh->mTetraCount;

    for (int i = 0; i < pCount; i++)
        adjList->push_back(new std::set<int>());

    //����� ����� ����������������
    for (int i = 0; i < vCount; i++)
    {
        int pNum1 = v[i].x;
        std::set<int>* tmp1 = adjList->at(pNum1);
        tmp1->insert(v[i].y);
        tmp1->insert(v[i].z);
        tmp1->insert(v[i].w);

        int pNum2 = v[i].y;
        std::set<int>* tmp2 = adjList->at(pNum2);
        tmp2->insert(v[i].x);
        tmp2->insert(v[i].z);
        tmp2->insert(v[i].w);

        int pNum3 = v[i].z;
        std::set<int>* tmp3 = adjList->at(pNum3);
        tmp3->insert(v[i].x);
        tmp3->insert(v[i].y);
        tmp3->insert(v[i].w);

        int pNum4 = v[i].w;
        std::set<int>* tmp4 = adjList->at(pNum4);
        tmp4->insert(v[i].x);
        tmp4->insert(v[i].y);
        tmp4->insert(v[i].z);
    }
}

//�� ������������ � ������ �����
float3 MeshSmooth::findIntersectionWithSTL(float3 point, MySTL * stl)
{
    int stlSize = stl->trigs.size() / 9;


    for (int i = 0; i < stlSize; i++)
    {
        float3 res = getIntersectionPoint(point,
            make_float3(stl->trigs[0], stl->trigs[1], stl->trigs[2]),
            make_float3(stl->trigs[3], stl->trigs[4], stl->trigs[5]),
            make_float3(stl->trigs[6], stl->trigs[7], stl->trigs[8]));
    }



    return make_float3(0.0f, 0.0f, 0.0f);
}

float3 MeshSmooth::getIntersectionPoint(float3 point, float3 p1, float3 p2, float3 p3)
{
    float3 A = p2 - p1;
    float3 B = p3 - p1;

    float3 direction = -cross(A, B);
    float3 N = cross(A, B);
    N = normalize(N);

    // Step 1: finding P
    // check if ray and plane are parallel?
    float NdotRayDirection = dot(N, direction);
    if (abs(NdotRayDirection) < 0.0001)
        return make_float3(INFINITY, INFINITY, INFINITY); // they are parallel so they don't intersect!

    float d = dot(N, p1);
    float t = -(dot(N, point) - d) / NdotRayDirection;
    float3 P = point + t * direction;
    // Step 2: inside-outside test //

    // vector perpendicular to triangle's plane
    // edge 1
    float3 edge1 = p2 - p1;
    float3 VP1 = P - p1;
    float3 C1 = cross(edge1, VP1);
    if (dot(N, C1) < 0)
        return make_float3(INFINITY, INFINITY, INFINITY); // P is on the right side
                  // edge 2
    float3 edge2 = p3 - p2;
    float3 VP2 = P - p2;
    float3 C2 = cross(edge2, VP2);
    if (dot(N, C2) < 0)
        return make_float3(INFINITY, INFINITY, INFINITY); // P is on the right side
                  // edge 3
    float3 edge3 = p1 - p3;
    float3 VP3 = P - p3;
    float3 C3 = cross(edge3, VP3);
    if (dot(N, C3) < 0)
        return make_float3(INFINITY, INFINITY, INFINITY); // P is on the right side
                  // P inside triangle

    return P;
}

inline float MeshSmooth::hasIntersection(float3 point, float3 dst, float3 p1, float3 p2, float3 p3)
{
    float3 direction = normalize(dst - point);

    float3 A = p2 - p1;
    float3 B = p3 - p1;

    float3 N = cross(A, B);
    N = normalize(N);

    // Step 1: finding P
    // check if ray and plane are parallel?
    float NdotRayDirection = dot(N, direction);
    if (abs(NdotRayDirection) < 0.0001)
        return 0; // they are parallel so they don't intersect!

    float d = dot(N, p1);
    float t = -(dot(N, point) - d) / NdotRayDirection;
    if (t < 0)
        return 0;// the triangle is behind
    float3 P = point + t * direction;
    // Step 2: inside-outside test //

    // vector perpendicular to triangle's plane
    // edge 1
    float3 edge1 = p2 - p1;
    float3 VP1 = P - p1;
    float3 C1 = cross(edge1, VP1);
    if (dot(N, C1) < 0)
        return -1; // P is on the right side
                   // edge 2
    float3 edge2 = p3 - p2;
    float3 VP2 = P - p2;
    float3 C2 = cross(edge2, VP2);
    if (dot(N, C2) < 0)
        return -1; // P is on the right side
                   // edge 3
    float3 edge3 = p1 - p3;
    float3 VP3 = P - p3;
    float3 C3 = cross(edge3, VP3);
    if (dot(N, C3) < 0)
        return -1; // P is on the right side
                   // P inside triangle
    return t;
}

void MeshSmooth::deletePoints(MyMesh * mesh, std::vector<int>* deletePoints)
{
}

std::vector<int2>* MeshSmooth::generateOldNewPairs(MyMesh * mesh, std::vector<int>* deleteNumbers, int & newpCount)
{
    deleteNumbers->push_back(INT_MAX);
    int oldpCount = mesh->mPointsCount;
    newpCount = 0;
    int posInDeleteList = 0;

    std::vector<int2>* result = new std::vector<int2>();

    //����� ������� ������ ��� ������, �� ����� ����������������, �.�. ���� ��� �����
    //���-�� ������� �����
    result->resize(oldpCount);

    //� ������� � ������ Push_back �� ������������
    for (int i = 0; i < oldpCount; i++)
    {
        if (i != deleteNumbers->at(posInDeleteList))
        {
            result->at(i) = (make_int2(i, newpCount));
            newpCount++;
        }
        else
        {
            result->at(i) = (make_int2(i, -1));
            posInDeleteList++;
        }
    }

    return result;
}
