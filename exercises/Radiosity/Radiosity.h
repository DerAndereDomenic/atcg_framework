#pragma once

#include <ATCG.h>

atcg::ref_ptr<atcg::TriMesh> solve_radiosity(const atcg::ref_ptr<atcg::TriMesh>& mesh, const torch::Tensor& emission);