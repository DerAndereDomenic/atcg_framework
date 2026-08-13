#pragma once

#include <Film/Film.h>
#include <Film/HDRFilmData.cuh>

namespace atcg
{
class ATCG_API HDRFilm : public Film
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

    // TODO
    virtual void updateData() override {}

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

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

private:
    torch::Tensor _accumulation_buffer;

    atcg::dref_ptr<HDRFilmData> _hdr_film_data;
};

}    // namespace atcg