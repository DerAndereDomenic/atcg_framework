#pragma once

#include <ATCG.h>

atcg::ref_ptr<atcg::TriMesh> solve_radiosity(const atcg::ref_ptr<atcg::TriMesh>& mesh, const torch::Tensor& emission);
atcg::ref_ptr<atcg::TriMesh> solve_radiosity_cpu(const atcg::ref_ptr<atcg::TriMesh>& mesh, const torch::Tensor& emission);
atcg::ref_ptr<atcg::TriMesh> solve_radiosity_gpu(const atcg::ref_ptr<atcg::TriMesh>& mesh, const torch::Tensor& emission);