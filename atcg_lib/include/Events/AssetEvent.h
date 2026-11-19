#pragma once

#include <Events/Event.h>
#include <Asset/Asset.h>
#include <sstream>

namespace atcg
{

/**
 * @brief Event that is sent when an asset is dragged from the asset panel and dropped onto the viewport.
 * The position is given in viewport pixel coordinates with the origin in the lower left corner, i.e., it can directly
 * be passed to Utils::pickEntity. Utils::handleViewportAssetDrop implements the default behavior.
 */
class AssetDroppedEvent : public Event
{
public:
    AssetDroppedEvent(AssetHandle handle, float x, float y) : _handle(handle), _x(x), _y(y) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "AssetDroppedEvent: " << (uint64_t)_handle << " (" << _x << ", " << _y << ")";
        return ss.str();
    }

    /**
     * @brief Get the handle of the dropped asset
     *
     * @return The asset handle
     */
    AssetHandle getHandle() const { return _handle; }

    /**
     * @brief Get the x coordinate of the drop position in viewport pixels
     *
     * @return The x coordinate
     */
    float getX() const { return _x; }

    /**
     * @brief Get the y coordinate of the drop position in viewport pixels (origin at the bottom)
     *
     * @return The y coordinate
     */
    float getY() const { return _y; }

    EVENT_CLASS_TYPE(AssetDropped)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)

private:
    AssetHandle _handle;
    float _x;
    float _y;
};
}    // namespace atcg
