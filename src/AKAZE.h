/**
 * @file AKAZE.h
 * @brief Main class for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
#include "AKAZEConfig.h"
#include "fed.h"
#include "cudaImage.h"
#include "cuda_akaze.h"

#include <utility>

/* ************************************************************************* */
namespace libAKAZECU {

    class Matcher {
	
    private:
	int maxnquery;
	unsigned char* descq_d;

	int maxntrain;
	unsigned char* desct_d;

	AkazeMatch* dmatches_d;
	AkazeMatch* dmatches_h;

	size_t pitch;
	
    public:
	Matcher() : maxnquery(0), descq_d(NULL), maxntrain(0), desct_d(NULL),
	    dmatches_d(0), dmatches_h(0), pitch(0) {}

	~Matcher();

	AkazeMat bfmatch_(AkazeMat desc_query, AkazeMat desc_train);
	
	void bfmatch(AkazeMat &desc_query, AkazeMat &desc_train,
		     std::vector<std::vector<AkazeMatch> > &dmatches);
	
    };


    class AKAZE {

  private:

    AKAZEOptions options_;                      ///< Configuration options for AKAZE
    std::vector<TEvolution> evolution_;         ///< Vector of nonlinear diffusion evolution

    /// FED parameters
    int ncycles_;                               ///< Number of cycles
    bool reordering_;                           ///< Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;                   ///< Vector of number of steps per cycle

    /// Matrices for the M-LDB descriptor computation
    AkazeMat descriptorSamples_;
    AkazeMat descriptorBits_;
    AkazeMat bitMask_;

    /// Computation times variables in ms
    AKAZETiming timing_;

    /// CUDA memory buffers
    float *cuda_memory;
    AkazeKeyPoint *cuda_points;
    AkazeKeyPoint *cuda_bufferpoints;
    AkazeMat cuda_desc;
    float* cuda_descbuffer;
    int* cuda_ptindices;
    CudaImage *cuda_images;
    std::vector<CudaImage> cuda_buffers;
    int nump;

  public:

    AKAZE(const AKAZEOptions& options);
    ~AKAZE();

    void Allocate_Memory_Evolution();

    int Create_Nonlinear_Scale_Space(const AkazeMat& img);

	AkazeMat Feature_Detection_();
    void Feature_Detection(std::vector<AkazeKeyPoint>& kpts);

    void Compute_Determinant_Hessian_Response();
    void Compute_Multiscale_Derivatives();

    void Find_Scale_Space_Extrema(std::vector<AkazeKeyPoint>& kpts);
    void Do_Subpixel_Refinement(std::vector<AkazeKeyPoint>& kpts);

    std::pair<AkazeMat, AkazeMat> Compute_Descriptors_Seq();
    void Compute_Descriptors(std::vector<AkazeKeyPoint>& kpts, AkazeMat& desc);

    AKAZETiming Get_Computation_Times() const {
      return timing_;
    }
  };

  /* ************************************************************************* */

  void setDefaultAKAZEOptions(AKAZEOptions& options);

  void generateDescriptorSubsample(AkazeMat& sampleList, AkazeMat& comparisons,
                                   int nbits, int pattern_size, int nchannels);

  inline void check_descriptor_limits(int& x, int& y, int width, int height);

  inline float gaussian(float x, float y, float sigma) {
    return expf(-(x*x+y*y)/(2.0f*sigma*sigma));
  }

  inline int fRound(float flt) {
    return (int)(flt+0.5f);
  }
}
