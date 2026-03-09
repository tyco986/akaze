#pragma once

#include "AKAZEConfig.h"
#include "cudaImage.h"

float *AllocBuffers(int width, int height, int num, int omax, int &maxpts, std::vector<CudaImage> &buffers, AkazeKeyPoint *&pts,
                    AkazeKeyPoint *&ptsbuffer, int *&ptindices, unsigned char *&desc, float *&descbuffer, CudaImage *&ims);
void InitCompareIndices();
void FreeBuffers(float *buffers);
double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var, int kernsize);
double Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly);
double Flow(CudaImage &img, CudaImage &flow, DIFFUSIVITY_TYPE type, float kcontrast);
double NLDStep(CudaImage &img,CudaImage &flow, CudaImage &temp, float stepsize);
double HalfSample(CudaImage &inimg, CudaImage &outimg);
double Copy(CudaImage &inimg, CudaImage &outimg);
double ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur, float perc, int nbins, float &contrast);
double HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly, int step);
double FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn, float border, float dthreshold, int scale, int octave, float size, AkazeKeyPoint *pts, int maxpts);
void FilterExtrema(AkazeKeyPoint *pts, AkazeKeyPoint *newpts, int *kptindices, int &nump);
void ClearPoints();
int GetPoints(std::vector<AkazeKeyPoint>& h_pts, AkazeKeyPoint *d_pts, int numPts);
void WaitCuda();
void GetDescriptors(AkazeMat &h_desc, AkazeMat &d_desc, int numPts);
double FindOrientation(AkazeKeyPoint *d_pts, std::vector<CudaImage> &h_imgs, CudaImage *d_imgs, int numPts);
double ExtractDescriptors(AkazeKeyPoint *d_pts, std::vector<CudaImage> &cuda_buffers, CudaImage *cuda_images, unsigned char* desc_h, float *vals_d, int patsize, int numPts);
void MatchDescriptors(AkazeMat &desc_query, AkazeMat &desc_train,
		      std::vector<std::vector<AkazeMatch> > &dmatches,
		      size_t pitch, 
		      unsigned char* descq_d, unsigned char* desct_d, AkazeMatch* dmatches_d, AkazeMatch* dmatches_h);
void MatchDescriptors(AkazeMat& desc_query, AkazeMat& desc_train, std::vector<std::vector<AkazeMatch> >& dmatches);
