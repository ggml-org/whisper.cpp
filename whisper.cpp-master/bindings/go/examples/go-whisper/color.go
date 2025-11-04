package main

import "fmt"

///////////////////////////////////////////////////////////////////////////////
// CONSTANTS

const (
	Reset     = "\033[0m"
	RGBPrefix = "\033[38;5;" // followed by RGB values in decimal format separated by colons
	RGBSuffix = "m"
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Colorize text with RGB values, from 0 to 23
func Colorize(text string, v int) string {
	// https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
	// Grayscale colors are in the range 232-255
	return RGBPrefix + fmt.Sprint(v%24+232) + RGBSuffix + text + Reset
}
