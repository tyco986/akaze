//=============================================================================
//
// AKAZE.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file AKAZE.cpp
 * @brief Main class for detecting and describing binary features in an
 * accelerated nonlinear scale space
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "AKAZE.h"
#include <cstdio>
#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;
using namespace libAKAZECU;

/* ---- timing helpers ---- */
static inline double tick_now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

/* ************************************************************************* */
void Matcher::bfmatch(AkazeMat &desc_query, AkazeMat &desc_train,
		      std::vector<std::vector<AkazeMatch> > &dmatches) {
    
    if (maxnquery < desc_query.rows) {
	if (descq_d) cudaFree(descq_d);
	if (dmatches_d) cudaFree(dmatches_d);
	cudaMallocPitch((void**)&descq_d, &pitch, 64, desc_query.rows);
	cudaMemset2D(descq_d, pitch, 0, 64, desc_query.rows);
	cudaMalloc((void**)&dmatches_d, desc_query.rows * 2 * sizeof(AkazeMatch));
	if (dmatches_h) delete [] dmatches_h;
	dmatches_h = new AkazeMatch[2 * desc_query.rows];
	maxnquery = desc_query.rows;
    }
    if (maxntrain < desc_train.rows) {
	if (desct_d) cudaFree(descq_d);
	cudaMallocPitch((void**)&desct_d, &pitch, 64, desc_train.rows);
	cudaMemset2DAsync(desct_d, pitch, 0, 64, desc_train.rows);
	maxntrain = desc_train.rows;
    }
    
    cudaMemcpy2DAsync(descq_d, pitch, desc_query.data, desc_query.cols,
		      desc_query.cols, desc_query.rows, cudaMemcpyHostToDevice);
    
    cudaMemcpy2DAsync(desct_d, pitch, desc_train.data, desc_train.cols,
		      desc_train.cols, desc_train.rows, cudaMemcpyHostToDevice);
    
    dim3 block(desc_query.rows);
    
    MatchDescriptors(desc_query, desc_train, dmatches, pitch,
		     descq_d, desct_d, dmatches_d, dmatches_h);
    
    cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(AkazeMatch),
	       cudaMemcpyDeviceToHost);
    
    dmatches.clear();
    for (int i = 0; i < desc_query.rows; ++i) {
	std::vector<AkazeMatch> tdmatch;
	tdmatch.push_back(dmatches_h[2 * i]);
	tdmatch.push_back(dmatches_h[2 * i + 1]);
	dmatches.push_back(tdmatch);
    }
    
}


AkazeMat Matcher::bfmatch_(AkazeMat desc_query, AkazeMat desc_train) {

    std::vector<std::vector<AkazeMatch> > dmatches_vec;

    bfmatch(desc_query, desc_train, dmatches_vec);
    
    AkazeMat dmatches_mat(dmatches_vec.size(), 8, AKAZE_32FC1);

    for (size_t i=0; i<dmatches_vec.size(); ++i) {
	float* mdata = (float*)&dmatches_mat.data[i*8*sizeof(float)];

	mdata[0] = dmatches_vec[i][0].queryIdx;
	mdata[1] = dmatches_vec[i][0].trainIdx;
	mdata[2] = 0.f;
	mdata[3] = dmatches_vec[i][0].distance;

	mdata[4] = dmatches_vec[i][1].queryIdx;
	mdata[5] = dmatches_vec[i][1].trainIdx;
	mdata[6] = 0.f;
	mdata[7] = dmatches_vec[i][1].distance;
    }
    
    return dmatches_mat;
}


Matcher::~Matcher() {
    if (descq_d) {
	cudaFree(descq_d);
    }
    if (desct_d) {
	cudaFree(desct_d);
    }
    if (dmatches_d) {
	cudaFree(dmatches_d);
    }
    if (dmatches_h) {
	delete [] dmatches_h;
    }
}



/* ************************************************************************* */
AKAZE::AKAZE(const AKAZEOptions& options) : options_(options) {
  ncycles_ = 0;
  reordering_ = true;

  if (options_.descriptor_size > 0 && options_.descriptor >= MLDB_UPRIGHT) {
    generateDescriptorSubsample(
        descriptorSamples_, descriptorBits_, options_.descriptor_size,
        options_.descriptor_pattern_size, options_.descriptor_channels);
  }

  Allocate_Memory_Evolution();
}

/* ************************************************************************* */
AKAZE::~AKAZE() {
  evolution_.clear();
  FreeBuffers(cuda_memory);
}

/* ************************************************************************* */
void AKAZE::Allocate_Memory_Evolution() {
  float rfactor = 0.0;
  int level_height = 0, level_width = 0;

  evolution_.reserve(options_.omax * options_.nsublevels);
  for (int i = 0; i <= options_.omax - 1; i++) {
    rfactor = 1.0 / pow(2.0f, i);
    level_height = (int)(options_.img_height * rfactor);
    level_width = (int)(options_.img_width * rfactor);

    if ((level_width < 80 || level_height < 40) && i != 0) {
      options_.omax = i;
      break;
    }

    for (int j = 0; j < options_.nsublevels; j++) {
      TEvolution step;
      AkazeSize size(level_width, level_height);
      step.Lx.create(size, AKAZE_32FC1);
      step.Ly.create(size, AKAZE_32FC1);
      step.Lxx.create(size, AKAZE_32FC1);
      step.Lxy.create(size, AKAZE_32FC1);
      step.Lyy.create(size, AKAZE_32FC1);
      step.Lt.create(size, AKAZE_32FC1);
      step.Ldet.create(size, AKAZE_32FC1);
      step.Lflow.create(size, AKAZE_32FC1);
      step.Lstep.create(size, AKAZE_32FC1);
      step.Lsmooth.create(size, AKAZE_32FC1);

      step.esigma = options_.soffset *
                    pow(2.0f, (float)(j) / (float)(options_.nsublevels) + i);
      step.sigma_size = fRound(step.esigma);
      step.etime = 0.5 * (step.esigma * step.esigma);
      step.octave = i;
      step.sublevel = j;
      evolution_.push_back(std::move(step));
    }
  }

  for (size_t i = 1; i < evolution_.size(); i++) {
    int naux = 0;
    vector<float> tau;
    float ttime = 0.0;
    ttime = evolution_[i].etime - evolution_[i - 1].etime;
    float tmax = 0.25;
    naux = fed_tau_by_process_time(ttime, 1, tmax, reordering_, tau);
    nsteps_.push_back(naux);
    tsteps_.push_back(tau);
    ncycles_++;
  }

  options_.ncudaimages = 4 * options_.nsublevels;
  unsigned char* _cuda_desc;
  cuda_memory = AllocBuffers(
      evolution_[0].Lt.cols, evolution_[0].Lt.rows, options_.ncudaimages,
      options_.omax, options_.maxkeypoints, cuda_buffers, cuda_bufferpoints,
      cuda_points, cuda_ptindices, _cuda_desc, cuda_descbuffer, cuda_images);
  cuda_desc = AkazeMat(options_.maxkeypoints, 61, AKAZE_8UC1, _cuda_desc);

}

/* ************************************************************************* */
int AKAZE::Create_Nonlinear_Scale_Space(const AkazeMat& img) {
  double t1 = 0.0, t2 = 0.0;

  if (evolution_.size() == 0) {
    cerr << "Error generating the nonlinear scale space!!" << endl;
    cerr << "Firstly you need to call AKAZE::Allocate_Memory_Evolution()"
         << endl;
    return -1;
  }

  t1 = tick_now();

  TEvolution& ev = evolution_[0];
  CudaImage& Limg = cuda_buffers[0];
  CudaImage& Lt = cuda_buffers[0];
  CudaImage& Lsmooth = cuda_buffers[1];
  CudaImage& Ltemp = cuda_buffers[2];

  Limg.h_data = (float*)img.data;
  Limg.Download();

  ContrastPercentile(Limg, Ltemp, Lsmooth, options_.kcontrast_percentile,
                     options_.kcontrast_nbins, options_.kcontrast);
  LowPass(Limg, Lt, Ltemp, options_.soffset * options_.soffset,
          2 * ceil((options_.soffset - 0.8) / 0.3) + 3);
  Copy(Lt, Lsmooth);

  Lt.h_data = (float*)ev.Lt.data;

  t2 = tick_now();
  timing_.kcontrast = 1000.0 * (t2 - t1);

  for (size_t i = 1; i < evolution_.size(); i++) {
    TEvolution& evn = evolution_[i];
    int num = options_.ncudaimages;
    CudaImage& Lt = cuda_buffers[evn.octave * num + 0 + 4 * evn.sublevel];
    CudaImage& Lsmooth = cuda_buffers[evn.octave * num + 1 + 4 * evn.sublevel];
    CudaImage& Lstep = cuda_buffers[evn.octave * num + 2];
    CudaImage& Lflow = cuda_buffers[evn.octave * num + 3];

    TEvolution& evo = evolution_[i - 1];
    CudaImage& Ltold = cuda_buffers[evo.octave * num + 0 + 4 * evo.sublevel];
    if (evn.octave > evo.octave) {
      HalfSample(Ltold, Lt);
      options_.kcontrast = options_.kcontrast * 0.75;
    } else
      Copy(Ltold, Lt);

    LowPass(Lt, Lsmooth, Lstep, 1.0, 5);
    Flow(Lsmooth, Lflow, options_.diffusivity, options_.kcontrast);

    for (int j = 0; j < nsteps_[i - 1]; j++) {
        float stepsize = tsteps_[i - 1][j] / (1 << 2 * evn.octave);
        NLDStep(Lt, Lflow, Lstep, tsteps_[i - 1][j]);
    }

    Lt.h_data = (float*)evn.Lt.data;
  }

  t2 = tick_now();
  timing_.scale = 1000.0 * (t2 - t1);
  
  return 0;
}


void kpvec2mat(std::vector<AkazeKeyPoint>& kpts, AkazeMat& _mat) {

    _mat = AkazeMat(kpts.size(), 7, AKAZE_32FC1);
    for (int i=0; i<(int)kpts.size(); ++i) {
        _mat.at<float>(i, 0) = kpts[i].pt.x;
        _mat.at<float>(i, 1) = kpts[i].pt.y;
        _mat.at<float>(i, 2) = kpts[i].size;
        _mat.at<float>(i, 3) = kpts[i].angle;
        _mat.at<float>(i, 4) = kpts[i].response;
        _mat.at<float>(i, 5) = (float)kpts[i].octave;
        _mat.at<float>(i, 6) = (float)kpts[i].class_id;
    }
}


AkazeMat AKAZE::Feature_Detection_() {

    std::vector<AkazeKeyPoint> kpts;

    this->Feature_Detection(kpts);

    AkazeMat mat;
    kpvec2mat(kpts,mat);

    return mat;
}


/* ************************************************************************* */
void AKAZE::Feature_Detection(std::vector<AkazeKeyPoint>& kpts) {
  double t1 = 0.0, t2 = 0.0;

  t1 = tick_now();

  int num = options_.ncudaimages;
  for (size_t i = 0; i < evolution_.size(); i++) {
    TEvolution& ev = evolution_[i];
    CudaImage& Lsmooth = cuda_buffers[ev.octave * num + 1 + 4 * ev.sublevel];
    CudaImage& Lx = cuda_buffers[ev.octave * num + 2 + 4 * ev.sublevel];
    CudaImage& Ly = cuda_buffers[ev.octave * num + 3 + 4 * ev.sublevel];

    float ratio = pow(2.0f, (float)evolution_[i].octave);
    int sigma_size_ =
        fRound(evolution_[i].esigma * options_.derivative_factor / ratio);
    HessianDeterminant(Lsmooth, Lx, Ly, sigma_size_);

    Lx.h_data = (float*)evolution_[i].Lx.data;
    Ly.h_data = (float*)evolution_[i].Ly.data;
  }
  t2 = tick_now();
  timing_.derivatives = 1000.0 * (t2 - t1);

  ClearPoints();
  for (size_t i = 0; i < evolution_.size(); i++) {
    TEvolution& ev = evolution_[i];
    TEvolution& evp = evolution_[(
        i > 0 && evolution_[i].octave == evolution_[i - 1].octave ? i - 1 : i)];
    TEvolution& evn =
        evolution_[(i < evolution_.size() - 1 &&
                            evolution_[i].octave == evolution_[i + 1].octave
                        ? i + 1
                        : i)];
    CudaImage& Ldet = cuda_buffers[ev.octave * num + 1 + 4 * ev.sublevel];
    CudaImage& LdetP = cuda_buffers[evp.octave * num + 1 + 4 * evp.sublevel];
    CudaImage& LdetN = cuda_buffers[evn.octave * num + 1 + 4 * evn.sublevel];

    float smax = 1.0f;
    if (options_.descriptor == SURF_UPRIGHT || options_.descriptor == SURF ||
        options_.descriptor == MLDB_UPRIGHT || options_.descriptor == MLDB)
      smax = 10.0 * sqrtf(2.0f);
    else if (options_.descriptor == MSURF_UPRIGHT ||
             options_.descriptor == MSURF)
      smax = 12.0 * sqrtf(2.0f);

    float ratio = pow(2.0f, (float)evolution_[i].octave);
    float size = evolution_[i].esigma * options_.derivative_factor;
    float border = smax * fRound(size / ratio);
    float thresh = std::max(options_.dthreshold, options_.min_dthreshold);

    FindExtrema(Ldet, LdetP, LdetN, border, thresh, i, evolution_[i].octave,
                size, cuda_points, options_.maxkeypoints);
  }

  FilterExtrema(cuda_points, cuda_bufferpoints, cuda_ptindices, nump);

  double t3 = tick_now();
  timing_.extrema = 1000.0 * (t3 - t2);
  timing_.detector = 1000.0 * (t3 - t1);
}


std::pair<AkazeMat, AkazeMat> AKAZE::Compute_Descriptors_Seq() {
    std::vector<AkazeKeyPoint> kptsvec;
    this->Feature_Detection(kptsvec);
    AkazeMat desc;
    this->Compute_Descriptors(kptsvec, desc);
    AkazeMat kpts;
    kpvec2mat(kptsvec, kpts);
    return {desc, kpts};
}


/* ************************************************************************* */
void AKAZE::Compute_Descriptors(std::vector<AkazeKeyPoint>& kpts,
                                AkazeMat& desc) {
  double t1 = 0.0, t2 = 0.0;

  t1 = tick_now();

  if (options_.descriptor < MLDB_UPRIGHT) {
    desc = AkazeMat::zeros(kpts.size(), 64, AKAZE_32FC1);
  } else {
    if (options_.descriptor_size == 0) {
      int t = (6 + 36 + 120) * options_.descriptor_channels;
      desc = AkazeMat::zeros(kpts.size(), (int)ceil(t / 8.), AKAZE_8UC1);
    } else {
      desc = AkazeMat::zeros(kpts.size(), (int)ceil(options_.descriptor_size / 8.),
                            AKAZE_8UC1);
    }
  }

  int pattern_size = options_.descriptor_pattern_size;

  switch (options_.descriptor) {
    case MLDB:
      FindOrientation(cuda_points, cuda_buffers, cuda_images, nump);
      GetPoints(kpts, cuda_points, nump);
      ExtractDescriptors(cuda_points, cuda_buffers, cuda_images,
                         cuda_desc.data, cuda_descbuffer, pattern_size, nump);
      GetDescriptors(desc, cuda_desc, nump);

      break;
    case SURF_UPRIGHT:
    case SURF:
    case MSURF_UPRIGHT:
    case MSURF:
    case MLDB_UPRIGHT:
      cout << "Descriptor not implemented\n";
  }

  t2 = tick_now();
  timing_.descriptor = 1000.0 * (t2 - t1);

  WaitCuda();
}


/* ************************************************************************* */
void libAKAZECU::generateDescriptorSubsample(AkazeMat& sampleList,
                                             AkazeMat& comparisons, int nbits,
                                             int pattern_size, int nchannels) {
  int ssz = 0;
  for (int i = 0; i < 3; i++) {
    int gz = (i + 2) * (i + 2);
    ssz += gz * (gz - 1) / 2;
  }
  ssz *= nchannels;

  assert(nbits <= ssz &&
            "descriptor size can't be bigger than full descriptor");

  AkazeMat_<int> fullM(ssz / nchannels, 5);
  for (size_t i = 0, c = 0; i < 3; i++) {
    int gdiv = i + 2;
    int gsz = gdiv * gdiv;
    int psz = ceil(2. * pattern_size / (float)gdiv);

    for (int j = 0; j < gsz; j++) {
      for (int k = j + 1; k < gsz; k++, c++) {
        fullM(c, 0) = i;
        fullM(c, 1) = psz * (j % gdiv) - pattern_size;
        fullM(c, 2) = psz * (j / gdiv) - pattern_size;
        fullM(c, 3) = psz * (k % gdiv) - pattern_size;
        fullM(c, 4) = psz * (k / gdiv) - pattern_size;
      }
    }
  }

  srand(1024);
  AkazeMat_<int> comps(nchannels * (int)ceil(nbits / (float)nchannels), 2);
  comps = 1000;

  int count = 0;
  size_t npicks = ceil(nbits / (float)nchannels);
  AkazeMat_<int> samples(29, 3);
  AkazeMat_<int> fullcopy = fullM.clone();
  samples = -1;

  for (size_t i = 0; i < npicks; i++) {
    size_t k = rand() % (fullM.rows - i);
    if (i < 6) {
      k = i;
    }

    bool n = true;

    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 1) &&
          samples(j, 2) == fullcopy(k, 2)) {
        n = false;
        comps(i * nchannels, 0) = nchannels * j;
        comps(i * nchannels + 1, 0) = nchannels * j + 1;
        comps(i * nchannels + 2, 0) = nchannels * j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 1);
      samples(count, 2) = fullcopy(k, 2);
      comps(i * nchannels, 0) = nchannels * count;
      comps(i * nchannels + 1, 0) = nchannels * count + 1;
      comps(i * nchannels + 2, 0) = nchannels * count + 2;
      count++;
    }

    n = true;
    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 3) &&
          samples(j, 2) == fullcopy(k, 4)) {
        n = false;
        comps(i * nchannels, 1) = nchannels * j;
        comps(i * nchannels + 1, 1) = nchannels * j + 1;
        comps(i * nchannels + 2, 1) = nchannels * j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 3);
      samples(count, 2) = fullcopy(k, 4);
      comps(i * nchannels, 1) = nchannels * count;
      comps(i * nchannels + 1, 1) = nchannels * count + 1;
      comps(i * nchannels + 2, 1) = nchannels * count + 2;
      count++;
    }

    AkazeMat tmp = fullcopy.row(k);
    fullcopy.row(fullcopy.rows - i - 1).copyTo(tmp);
  }

  sampleList = samples.rowRange(0, count).clone();
  comparisons = comps.rowRange(0, nbits).clone();
}

/* ************************************************************************* */
void libAKAZECU::check_descriptor_limits(int& x, int& y, int width,
                                         int height) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x > width - 1) x = width - 1;
  if (y > height - 1) y = height - 1;
}
