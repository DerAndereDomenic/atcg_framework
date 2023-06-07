#pragma once

#include <DataStructure/Mesh.h>

atcg::ref_ptr<atcg::Mesh> solve_radiosity(const atcg::ref_ptr<atcg::Mesh>& mesh, const Eigen::MatrixX3f& emission);