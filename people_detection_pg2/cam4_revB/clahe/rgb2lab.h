// librgb2lab
// source: https://github.com/jamadagni/rgb2lab/blob/master/rgb2lab.h
// ==========
//
// Convert color values from RGB to/from CIE LAB/LCH
// for sRGB gamut, D65 illuminant, 2° observer
//
// Copyright (C) 2015, Shriramana Sharma, samjnaa-at-gmail-dot-com
//
// Use, modification and distribution are permitted subject to the
// "BSD-2-Clause"-type license stated in the accompanying file LICENSE.txt

#ifdef __cplusplus
extern "C" {
#endif

typedef union
{
    double data[3];
    struct { double r, g, b; };
    struct { double L, A, B; };
    struct { double l, c, h; };
} DoubleTriplet;

DoubleTriplet rgbFromLab(DoubleTriplet lab);
DoubleTriplet labFromRgb(DoubleTriplet rgb);
DoubleTriplet lchFromLab(DoubleTriplet lab);
DoubleTriplet labFromLch(DoubleTriplet lch);
DoubleTriplet lchFromRgb(DoubleTriplet rgb);
DoubleTriplet rgbFromLch(DoubleTriplet lch);

#ifdef __cplusplus
}
#endif
