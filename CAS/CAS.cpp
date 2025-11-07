#include "CAS.h"
#include <math.h>
#include <xmmintrin.h>

static PF_Err About (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	suites.ANSICallbacksSuite1()->sprintf(	out_data->return_msg,
											"%s v%d.%d\r%s",
											STR(StrID_Name),
											MAJOR_VERSION,
											MINOR_VERSION,
											STR(StrID_Description));
	return PF_Err_NONE;
}

static PF_Err GlobalSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	out_data->my_version = PF_VERSION(	MAJOR_VERSION,
										MINOR_VERSION,
										BUG_VERSION,
										STAGE_VERSION,
										BUILD_VERSION);

	out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;	// just 16bpc, not 32bpc

	return PF_Err_NONE;
}

static PF_Err ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err		err		= PF_Err_NONE;
	PF_ParamDef	def;

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Sharpening_Param_Name),
							CAS_SHARPENING_MIN,
							CAS_SHARPENING_MAX,
							CAS_SHARPENING_MIN,
							CAS_SHARPENING_MAX,
							CAS_SHARPENING_DFLT,
							PF_Precision_HUNDREDTHS,
							0,
							0,
							SHARPENING_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Contrast_Param_Name),
							CAS_CONTRAST_MIN,
							CAS_CONTRAST_MAX,
							CAS_CONTRAST_MIN,
							CAS_CONTRAST_MAX,
							CAS_CONTRAST_DFLT,
							PF_Precision_HUNDREDTHS,
							0,
							0,
							CONTRAST_DISK_ID);

	out_data->num_params = CAS_NUM_PARAMS;

	return err;
}

typedef struct {
	float peak;
	float sharpening_coeff;
	PF_EffectWorld* input;
	A_long width;
	A_long height;
} CASParams, *CASParamsP, **CASParamsH;

// Fast reciprocal square root approximation using SSE intrinsic
inline __m128 simd_fast_rsqrt(__m128 x) {
    return _mm_rsqrt_ps(x); // Standard SSE intrinsic for 1/sqrt(x)
}

// SIMD versions of float3 functions
inline __m128 simd_set3(float val) {
	return _mm_set_ps(0.0f, val, val, val); // Set R, G, B to val, A to 0 (dummy)
}

inline __m128 simd_make3(float x, float y, float z) {
	return _mm_set_ps(0.0f, z, y, x); // Set R, G, B to x, y, z, A to 0 (dummy)
}

// Note: SSE fmin/fmax handle NaN differently than C fminf/fmaxf. We assume no NaN inputs.
inline __m128 simd_fmin(__m128 a, __m128 b) {
	return _mm_min_ps(a, b);
}

inline __m128 simd_fmax(__m128 a, __m128 b) {
	return _mm_max_ps(a, b);
}

inline __m128 simd_add(__m128 a, __m128 b) {
	return _mm_add_ps(a, b);
}

inline __m128 simd_sub(__m128 a, __m128 b) {
	return _mm_sub_ps(a, b);
}

inline __m128 simd_mul(__m128 a, __m128 b) {
	return _mm_mul_ps(a, b);
}

inline __m128 simd_div(__m128 a, __m128 b) {
	return _mm_div_ps(a, b);
}

// Inline reciprocal using SSE
inline __m128 simd_rcp(__m128 a) {
	return _mm_rcp_ps(a);
}

// Inline saturate using SSE
inline __m128 simd_saturate(__m128 a) {
	__m128 zero = _mm_setzero_ps();
	__m128 one = _mm_set1_ps(1.0f);
	return _mm_min_ps(_mm_max_ps(a, zero), one);
}

// Inline lerp using SSE
inline __m128 simd_lerp(__m128 a, __m128 b, __m128 t) {
	return simd_add(a, simd_mul(t, simd_sub(b, a)));
}

static void CASPixel8(
	PF_Pixel8* a, PF_Pixel8* b, PF_Pixel8* c,
	PF_Pixel8* d, PF_Pixel8* e, PF_Pixel8* f,
	PF_Pixel8* g, PF_Pixel8* h, PF_Pixel8* i,
	PF_Pixel8* outPixel,
	float peak,
	float sharpening_coeff)
{
	const float r = 1.0f / 255.0f;
	__m128 b_val = simd_make3(b->red * r, b->green * r, b->blue * r);
	__m128 d_val = simd_make3(d->red * r, d->green * r, d->blue * r);
	__m128 e_val = simd_make3(e->red * r, e->green * r, e->blue * r);
	__m128 f_val = simd_make3(f->red * r, f->green * r, f->blue * r);
	__m128 h_val = simd_make3(h->red * r, h->green * r, h->blue * r);
	__m128 a_val = simd_make3(a->red * r, a->green * r, a->blue * r);
	__m128 c_val = simd_make3(c->red * r, c->green * r, c->blue * r);
	__m128 g_val = simd_make3(g->red * r, g->green * r, g->blue * r);
	__m128 i_val = simd_make3(i->red * r, i->green * r, i->blue * r);

	// Soft min and max (as in CAS.fx) - using SIMD expanded logic
	// a b c
	// d(e)f
	// g h i
	// mnRGB = fminf(fminf(fminf(fminf(d_val, e_val), fminf(f_val, b_val)), h_val), fminf(fminf(a_val, c_val), fminf(g_val, i_val)));
	__m128 temp_mn1 = simd_fmin(simd_fmin(d_val, e_val), simd_fmin(f_val, b_val));
	__m128 temp_mn2 = simd_fmin(simd_fmin(a_val, c_val), simd_fmin(g_val, i_val));
	__m128 mnRGB = simd_fmin(simd_fmin(temp_mn1, h_val), temp_mn2);

	// mxRGB = fmaxf(fmaxf(fmaxf(fmaxf(d_val, e_val), fmaxf(f_val, b_val)), h_val), fmaxf(fmaxf(a_val, c_val), fmaxf(g_val, i_val)));
	__m128 temp_mx1 = simd_fmax(simd_fmax(d_val, e_val), simd_fmax(f_val, b_val));
	__m128 temp_mx2 = simd_fmax(simd_fmax(a_val, c_val), simd_fmax(g_val, i_val));
	__m128 mxRGB = simd_fmax(simd_fmax(temp_mx1, h_val), temp_mx2);

	// Smooth minimum distance to signal limit divided by smooth max.
	__m128 rcpMRGB = simd_rcp(mxRGB);
	// Calculate fminf(mnRGB, (2.0f - mxRGB)) and multiply by rcpMRGB
	__m128 two_minus_mx = simd_sub(simd_set3(2.0f), mxRGB);
	__m128 temp_min_dist = simd_fmin(mnRGB, two_minus_mx);
	__m128 ampRGB_pre_saturate = simd_mul(temp_min_dist, rcpMRGB);
	// Saturate the values
	__m128 ampRGB_saturated = simd_saturate(ampRGB_pre_saturate);
	// Shaping amount of sharpening.
	__m128 ampRGB = simd_fast_rsqrt(ampRGB_saturated); // Use rsqrt instead of sqrt as in CAS.fx

	// Use pre-computed peak instead of recalculating
	__m128 amp_peak = simd_mul(ampRGB, simd_set3(peak));
	__m128 rcp_amp_peak = simd_rcp(amp_peak);
	// Use neg inline:
	__m128 neg_one = simd_set3(-1.0f);
	__m128 wRGB = simd_mul(neg_one, rcp_amp_peak);

	// Calculate (4.0f * wRGB) + 1.0f
	__m128 four_w_plus_one = simd_add(simd_mul(simd_set3(4.0f), wRGB), simd_set3(1.0f));
	__m128 rcpWeightRGB = simd_rcp(four_w_plus_one);

	// Filter shape: 0 w 0, w 1 w, 0 w 0
	// Calculate (b_val + d_val + f_val + h_val) * wRGB + e_val
	__m128 window_sum = simd_add(simd_add(b_val, d_val), simd_add(f_val, h_val));
	__m128 filtered_val = simd_add(simd_mul(window_sum, wRGB), e_val);
	// Calculate outColor = filtered_val * rcpWeightRGB
	__m128 outColor_pre_saturate = simd_mul(filtered_val, rcpWeightRGB);
	// Saturate the result
	__m128 outColor = simd_saturate(outColor_pre_saturate);

	// Use pre-computed sharpening coefficient
	// Calculate lerp(e_val, outColor, sharpening_coeff)
	__m128 t_scalar = simd_set3(sharpening_coeff);
	__m128 finalColor = simd_lerp(e_val, outColor, t_scalar);

	// Convert back to pixel format using fast clamp
	// Extract R, G, B from __m128 (stored as r, g, b, dummy)
	float results[4];
	_mm_store_ps(results, finalColor); // Store results to array

	// Manually clamp and convert to A_u_char
	// Using simple cast and manual clamp is often faster than std::clamp or math functions for this range
	A_u_char r_u_char = (results[0] > 1.0f) ? 255 : (results[0] < 0.0f) ? 0 : (A_u_char)(results[0] * 255.0f);
	A_u_char g_u_char = (results[1] > 1.0f) ? 255 : (results[1] < 0.0f) ? 0 : (A_u_char)(results[1] * 255.0f);
	A_u_char b_u_char = (results[2] > 1.0f) ? 255 : (results[2] < 0.0f) ? 0 : (A_u_char)(results[2] * 255.0f);

	outPixel->red = r_u_char;
	outPixel->green = g_u_char;
	outPixel->blue = b_u_char;
	outPixel->alpha = e->alpha;
}

// Iterate function for 8-bit processing
static PF_Err CASIterate8(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8* inP,
	PF_Pixel8* outP)
{
	PF_Err		err = PF_Err_NONE;

	CASParams* params = reinterpret_cast<CASParams*>(refcon);

	if (params) {
		PF_EffectWorld* input = params->input;
		A_long width = params->width;
		A_long height = params->height;

		if (xL == 0 || xL == width - 1 || yL == 0 || yL == height - 1) {
			*outP = *inP;
			return err;
		}

		// Get input pixel data directly from the input world
		PF_Pixel8* inputPixels = (PF_Pixel8*)input->data;

		// Define the 3x3 neighborhood
		PF_Pixel8* a = &inputPixels[(yL - 1) * width + (xL - 1)]; // top-left
		PF_Pixel8* b = &inputPixels[(yL - 1) * width + xL];     // top
		PF_Pixel8* c = &inputPixels[(yL - 1) * width + (xL + 1)]; // top-right
		PF_Pixel8* d = &inputPixels[yL * width + (xL - 1)]; // left
		PF_Pixel8* e = &inputPixels[yL * width + xL];     // center (same as inP)
		PF_Pixel8* f = &inputPixels[yL * width + (xL + 1)]; // right
		PF_Pixel8* g = &inputPixels[(yL + 1) * width + (xL - 1)]; // bottom-left
		PF_Pixel8* h = &inputPixels[(yL + 1) * width + xL];     // bottom
		PF_Pixel8* i = &inputPixels[(yL + 1) * width + (xL + 1)]; // bottom-right

		CASPixel8(a, b, c, d, e, f, g, h, i, outP, params->peak, params->sharpening_coeff);
	}
	else {
		// If no refcon, just copy the input to output
		*outP = *inP;
	}

	return err;
}

static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err				err		= PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	PF_FpLong sharpeningF = params[CAS_SHARPENING]->u.fs_d.value;
	PF_FpLong contrastF = params[CAS_CONTRAST]->u.fs_d.value;
	PF_EffectWorld* input = &params[CAS_INPUT]->u.ld;

	A_long width = input->width;
	A_long height = input->height;

	// Initialize output with the same dimensions
	ERR(PF_COPY(&params[CAS_INPUT]->u.ld, output, NULL, NULL));

	CASParams casParams;
	casParams.peak = -3.0f * (float)contrastF / 100.0f + 8.0f;
	casParams.sharpening_coeff = (float)sharpeningF / 100.0f;
	casParams.input = input;
	casParams.width = width;
	casParams.height = height;

	if (PF_WORLD_IS_DEEP(output)) {
		return err;
	} else {
		// Process 8-bit image using iterate8 suite
		ERR(suites.Iterate8Suite2()->iterate(
			in_data,
			0,               // progress base
			height,          // progress final
			input,           // source world
			&output->extent_hint, // rectangle to iterate over
			(void*)&casParams,    // refcon
			CASIterate8,          // pixel processing function
			output               // destination world
		));
	}

	return err;
}


extern "C" DllExport
PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		"Contrast Adaptive Sharpening", // Name
		"ADBE CAS", // Match Name
		"Blur & Sharpen", // Category
		AE_RESERVED_INFO, // Reserved Info
		"EffectMain",	// Entry point
		"https://www.adobe.com");	// support URL

	return result;
}


PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;

	try {
		switch (cmd) {
			case PF_Cmd_ABOUT:

				err = About(in_data,
							out_data,
							params,
							output);
				break;

			case PF_Cmd_GLOBAL_SETUP:

				err = GlobalSetup(	in_data,
									out_data,
									params,
									output);
				break;

			case PF_Cmd_PARAMS_SETUP:

				err = ParamsSetup(	in_data,
									out_data,
									params,
									output);
				break;

			case PF_Cmd_RENDER:

				err = Render(	in_data,
								out_data,
								params,
								output);
				break;
		}
	}
	catch(PF_Err &thrown_err){
		err = thrown_err;
	}
	return err;
}