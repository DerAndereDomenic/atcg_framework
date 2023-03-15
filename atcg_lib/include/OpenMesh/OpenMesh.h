#pragma once

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Utils/PropertyManager.hh>

#include <OpenMesh/GLMTraits.h>

namespace atcg
{
typedef OpenMesh::TriMesh_ArrayKernelT<GLMTraits> TriMesh;
}