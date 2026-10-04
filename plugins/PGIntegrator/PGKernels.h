#pragma once

#include <Core/Memory.h>
#include "vMF.h"

namespace atcg
{
void updateMixtures(const atcg::DeviceBuffer<vMFMixture<5>>& vmf_mixtures);
}