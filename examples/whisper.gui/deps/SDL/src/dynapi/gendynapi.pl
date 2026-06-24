#!/usr/bin/perl -w

#  Simple DirectMedia Layer
#  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>
#
#  This software is provided 'as-is', without any express or implied
#  warranty.  In no event will the authors be held liable for any damages
#  arising from the use of this software.
#
#  Permission is granted to anyone to use this software for any purpose,
#  including commercial applications, and to alter it and redistribute it
#  freely, subject to the following restrictions:
#
#  1. The origin of this software must not be misrepresented; you must not
#     claim that you wrote the original software. If you use this software
#     in a product, an acknowledgment in the product documentation would be
#     appreciated but is not required.
#  2. Altered source versions must be plainly marked as such, and must not be
#     misrepresented as being the original software.
#  3. This notice may not be removed or altered from any source distribution.

# WHAT IS THIS?
#  When you add a public API to SDL, please run this script, make sure the
#  output looks sane (git diff, it adds to existing files), and commit it.
#  It keeps the dynamic API jump table operating correctly.

# If you wanted this to be readable, you shouldn't have used perl.

use warnings;
use strict;
use File::Basename;

chdir(dirname(__FILE__) . '/../..');
my $sdl_dynapi_procs_h = "src/dynapi/SDL_dynapi_procs.h";
my $sdl_dynapi_overrides_h = "src/dynapi/SDL_dynapi_overrides.h";
my $sdl2_exports = "src/dynapi/SDL2.exports";

my %existing = ();
if (-f $sdl_dynapi_procs_h) {
    open(SDL_DYNAPI_PROCS_H, '<', $sdl_dynapi_procs_h) or die("Can't open $sdl_dynapi_procs_h: $!\n");
    while (<SDL_DYNAPI_PROCS_H>) {
        if (/\ASDL_DYNAPI_PROC\(.*?,(.*?),/) {
            $existing{$1} = 1;
        }
    }
    close(SDL_DYNAPI_PROCS_H)
}

open(SDL_DYNAPI_PROCS_H, '>>', $sdl_dynapi_procs_h) or die("Can't open $sdl_dynapi_procs_h: $!\n");
open(SDL_DYNAPI_OVERRIDES_H, '>>', $sdl_dynapi_overrides_h) or die("Can't open $sdl_dynapi_overrides_h: $!\n");
open(SDL2_EXPORTS, '>>', $sdl2_exports) or die("Can't open $sdl2_exports: $!\n");

# Ordered for reproducible builds
opendir(HEADERS, 'include') or die("Can't open include dir: $!\n");
my @entries = readdir(HEADERS);
closedir(HEADERS);
# Sort the entries
@entries = sort @entries;
foreach my $d (@entries) {
    next if not $d =~ /\.h\Z/;
    my $header = "include/$d";
    open(HEADER, '<', $header) or die("Can't open $header: $!\n");
    while (<HEADER>) {
        chomp;
        next if not /\A\s*extern\s+(SDL_DEPRECATED\s+|)DECLSPEC/;
        my $decl = "$_ ";
        if (not $decl =~ /\)\s*;/) {
            while (<HEADER>) {
                chomp;
                s/\A\s+//;
                s/\s+\Z//;
                $decl .= "$_ ";
                last if /\)\s*;/;
            }
        }

        $decl =~ s/\s+\Z//;
        #print("DECL: [$decl]\n");

        if ($decl =~ /\A\s*extern\s+(SDL_DEPRECATED\s+|)DECLSPEC\s+(const\s+|)(unsigned\s+|)(.*?)\s*(\*?)\s*SDLCALL\s+(.*?)\s*\((.*?)\);/) {
            my $rc = "$2$3$4$5";
            my $fn = $6;

            next if $existing{$fn};   # already slotted into the jump table.

            my @params = split(',', $7);

            #print("rc == '$rc', fn == '$fn', params == '$params'\n");

            my $retstr = ($rc eq 'void') ? '' : 'return';
            my $paramstr = '(';
            my $argstr = '(';
            my $i = 0;
            foreach (@params) {
                my $str = $_;
                $str =~ s/\A\s+//;
                $str =~ s/\s+\Z//;
                #print("1PARAM: $str\n");
                if ($str eq 'void') {
                    $paramstr .= 'void';
                } elsif ($str eq '...') {
                    if ($i > 0) {
                        $paramstr .= ', ';
                    }
                    $paramstr .= $str;
                } elsif ($str =~ /\A\s*((const\s+|)(unsigned\s+|)([a-zA-Z0-9_]*)\s*([\*\s]*))\s*(.*?)\Z/) {
                    #print("PARSED: [$1], [$2], [$3], [$4], [$5]\n");
                    my $type = $1;
                    my $var = $6;
                    $type =~ s/\A\s+//;
                    $type =~ s/\s+\Z//;
                    $var =~ s/\A\s+//;
                    $var =~ s/\s+\Z//;
                    $type =~ s/\s*\*\Z/*/g;
                    $type =~ s/\s*(\*+)\Z/ $1/;
                    #print("SPLIT: ($type, $var)\n");
                    my $var_array_suffix = "";
                    # parse array suffix
                    if ($var =~ /\A.*(\[.*\])\Z/) {
                        #print("PARSED ARRAY SUFFIX: [$1] of '$var'\n");
                        $var_array_suffix = $1;
                    }
                    my $name = chr(ord('a') + $i);
                    if ($i > 0) {
                        $paramstr .= ', ';
                        $argstr .= ',';
                    }
                    my $spc = ($type =~ /\*\Z/) ? '' : ' ';
                    $paramstr .= "$type$spc$name$var_array_suffix";
                    $argstr .= "$name";
                }
                $i++;
            }

            $paramstr = '(void' if ($i == 0);  # Just to make this consistent.

            $paramstr .= ')';
            $argstr .= ')';

            print("NEW: $decl\n");
            print SDL_DYNAPI_PROCS_H "SDL_DYNAPI_PROC($rc,$fn,$paramstr,$argstr,$retstr)\n";
            print SDL_DYNAPI_OVERRIDES_H "#define $fn ${fn}_REAL\n";
            print SDL2_EXPORTS "++'_${fn}'.'SDL2.dll'.'${fn}'\n";
        } else {
            print("Failed to parse decl [$decl]!\n");
        }
    }
    close(HEADER);
}

close(SDL_DYNAPI_PROCS_H);
close(SDL_DYNAPI_OVERRIDES_H);
close(SDL2_EXPORTS);

# vi: set ts=4 sw=4 expandtab:
