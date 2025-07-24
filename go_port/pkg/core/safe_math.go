// Package core provides safe mathematical operations for aerospace-grade precision calculations.
package core

import "math"

// SafeDiv performs division with protection against infinite and NaN values
func SafeDiv(numerator, denominator float64) float64 {
	if denominator == 0 || math.IsInf(denominator, 0) || math.IsNaN(denominator) {
		return 0.0
	}
	if math.IsInf(numerator, 0) || math.IsNaN(numerator) {
		return 0.0
	}

	result := numerator / denominator
	if math.IsInf(result, 0) || math.IsNaN(result) {
		return 0.0
	}
	return result
}

// SafeDivInt performs division between float and int with protection against infinite values
func SafeDivInt(numerator float64, denominator int64) float64 {
	if denominator == 0 {
		return 0.0
	}
	return SafeDiv(numerator, float64(denominator))
}

// SafeRatio calculates a ratio with bounds checking
func SafeRatio(numerator, denominator float64) float64 {
	ratio := SafeDiv(numerator, denominator)
	// Clamp extremely large ratios to prevent JSON marshaling issues
	if ratio > 1e10 {
		return 1e10
	}
	if ratio < -1e10 {
		return -1e10
	}
	return ratio
}

// ValidateFloat checks if a float64 value is valid for JSON marshaling
func ValidateFloat(value float64) float64 {
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return 0.0
	}
	// Clamp extremely large values
	if value > 1e15 {
		return 1e15
	}
	if value < -1e15 {
		return -1e15
	}
	return value
}

// SafePercent calculates a percentage with bounds
func SafePercent(part, total float64) float64 {
	if total == 0 {
		return 0.0
	}
	percent := SafeDiv(part*100, total)
	return ValidateFloat(percent)
}

// SafeCoefficientOfVariation calculates CV with protection against edge cases
func SafeCoefficientOfVariation(stdDev, mean float64) float64 {
	if mean == 0 || stdDev == 0 {
		return 0.0
	}
	cv := SafeDiv(stdDev, math.Abs(mean))
	return ValidateFloat(cv)
}
