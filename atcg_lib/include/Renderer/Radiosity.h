#pragma once

#include <DataStructure/Mesh.h>

namespace atcg
{
atcg::ref_ptr<Mesh> solve_radiosity(const atcg::ref_ptr<Mesh>& mesh, const Eigen::MatrixX3f& emission);
}