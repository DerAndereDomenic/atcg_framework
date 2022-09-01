#include <Core/LayerStack.h>

namespace atcg {

	LayerStack::~LayerStack()
	{
		for (Layer* layer : _layers)
		{
			layer->onDetach();
			delete layer;
		}
	}

	void LayerStack::pushLayer(Layer* layer)
	{
		_layers.emplace(_layers.begin() + _layer_insert_index, layer);
		_layer_insert_index++;
	}

	void LayerStack::popLayer(Layer* layer)
	{
		auto it = std::find(_layers.begin(), _layers.begin() + _layer_insert_index, layer);
		if (it != _layers.begin() + _layer_insert_index)
		{
			layer->onDetach();
			_layers.erase(it);
			_layer_insert_index--;
		}
	}

}