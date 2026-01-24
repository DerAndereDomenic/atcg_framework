#pragma once

#include <Film/Film.h>
#include <Film/HDRFilmData.cuh>

namespace atcg
{
class HDRFilm : public Film
{
public:
    /**
     * @brief Default Constructor
     */
    HDRFilm();

    /**
     * @brief Constructor.
     * Requires the following dictionary entries:
     * - width: uint32_t
     * - height: uint32_t
     *
     * @param dict The parameters
     */
    HDRFilm(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~HDRFilm();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

    /**
     * @brief Resize the film
     *
     * @param width The new width
     * @param height The new height
     */
    virtual void resize(uint32_t width, uint32_t height) override;

    /**
     * @brief Develop the film
     */
    virtual torch::Tensor develop() const override;

    /**
     * @brief Clear the film
     */
    virtual void clear() override;

    /**
     * @brief Get the accumulation buffer
     *
     * @return The accumulation buffer
     */
    ATCG_INLINE torch::Tensor getAccumulationBuffer() const { return _accumulation_buffer; }

    /**
     * @brief Get the data buffer
     *
     * @return The data buffer
     */
    ATCG_INLINE atcg::dref_ptr<HDRFilmData> getDataBuffer() const { return _hdr_film_data; }

private:
    torch::Tensor _accumulation_buffer;

    atcg::dref_ptr<HDRFilmData> _hdr_film_data;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(HDRFilm);
}    // namespace atcg