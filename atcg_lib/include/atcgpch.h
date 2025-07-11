#pragma once

// STL
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <cstdint>
#include <numeric>
#include <utility>
#include <functional>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <array>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

// Engine
#include <Core/API.h>
#include <Core/CUDA.h>
#include <Core/glm.h>
#include <Core/Input.h>
#include <Core/Layer.h>
#include <Core/LayerStack.h>
#include <Core/Log.h>
#include <Core/Memory.h>
#include <Core/UUID.h>
#include <Events/Event.h>
#include <Events/KeyEvent.h>
#include <Events/MouseEvent.h>
#include <Events/WindowEvent.h>

#include <torch/types.h>