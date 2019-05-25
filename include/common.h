#pragma once
// Header for often includes //
#include <memory>
#include <vector>
// #include <iosfwd>

// TypeDefs //
typedef int Point4i[4];

struct vec3d {
    float x, y, z;
};
struct vec3i {
    int x, y, z;
};
struct vec4i {
    int x, y, z, w;
};

enum GRAIN_STATUS {
    GRAIN_ERR_NONE = 0,
    GRAIN_ERR_WRONG_PARAMETER = -1
};

int version();
