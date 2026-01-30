#pragma once

#include <Core/Platform.h>
#include <Core/OptixComponent.h>
#include <DataStructure/Dictionary.h>
#include <Film/FilmVPtrTable.cuh>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
class Film : public OptixComponent
{
public:
    /**
     * @brief Constructor
     */
    Film() = default;

    /**
     * @brief Construct a film with arbitrary parameters
     *
     * @param dict Dictionary holding the parameters
     */
    Film(const atcg::Dictionary& dict) {}

    /**
     * @brief Destructor
     */
    virtual ~Film() {}

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

    /**
     * @brief Resize the film
     *
     * @param width The new width
     * @param height The new height
     */
    virtual void resize(uint32_t width, uint32_t height) = 0;

    /**
     * @brief Develop the film
     */
    virtual torch::Tensor develop() const = 0;

    /**
     * @brief Clear the film
     */
    virtual void clear() = 0;

    /**
     * @brief Get the VPtrTable
     *
     * @return The VPtrTable
     */
    inline const FilmVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief Get the width of the film
     *
     * @return The width
     */
    ATCG_INLINE uint32_t getWidth() const { return _width; }

    /**
     * @brief Get the height of the film
     *
     * @return The height
     */
    ATCG_INLINE uint32_t getHeight() const { return _height; }

protected:
    uint32_t _width  = 0;
    uint32_t _height = 0;
    atcg::dref_ptr<FilmVPtrTable> _vptr_table;
};
}    // namespace atcg