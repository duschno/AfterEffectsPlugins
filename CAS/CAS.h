/*
	CAS.h
*/

#pragma once

#ifndef CAS_H
#define CAS_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1	// make sure we get 16bpc pixels; 
								// AE_Effect.h checks for this.

#include "AEConfig.h"

#ifdef AE_OS_WIN
	typedef unsigned short PixelType;
	#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "CAS_Strings.h"

/* Versioning information */

#define	MAJOR_VERSION	1
#define	MINOR_VERSION	1
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


/* Parameter defaults */

#define	CAS_SHARPENING_MIN		0
#define	CAS_SHARPENING_MAX		100
#define	CAS_SHARPENING_DFLT		100

#define	CAS_CONTRAST_MIN		0
#define	CAS_CONTRAST_MAX		100
#define	CAS_CONTRAST_DFLT		0

enum {
	CAS_INPUT = 0,
	CAS_SHARPENING,
	CAS_CONTRAST,
	CAS_NUM_PARAMS
};

enum {
	SHARPENING_DISK_ID = 1,
	CONTRAST_DISK_ID,
};

typedef struct SharpeningInfo{
	PF_FpLong	sharpeningF;
} SharpeningInfo, *SharpeningInfoP, **SharpeningInfoH;

typedef struct ContrastInfo {
	PF_FpLong	contrastF;
} ContrastInfo, * ContrastInfoP, ** ContrastInfoH;


extern "C" {

	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#endif // CAS_H