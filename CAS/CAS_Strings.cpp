#include "CAS.h"

typedef struct {
	A_u_long	index;
	A_char		str[256];
} TableString;



TableString		g_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						"Contrast Adaptive Sharpening",
	StrID_Description,				"A port of Reshade implementation of\nAMD FidelityFX Contrast Adaptive Sharpening (CAS)",
	StrID_Sharpening_Param_Name,	"Sharpening",
	StrID_Contrast_Param_Name,		"Contrast",
};


char	*GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}
	