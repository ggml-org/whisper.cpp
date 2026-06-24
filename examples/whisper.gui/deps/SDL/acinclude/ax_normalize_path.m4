# ===========================================================================
#    https://www.gnu.org/software/autoconf-archive/ax_normalize_path.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_NORMALIZE_PATH(VARNAME, [REFERENCE_STRING])
#
# DESCRIPTION
#
#   Perform some cleanups on the value of $VARNAME (interpreted as a path):
#
#     - empty paths are changed to '.'
#     - trailing slashes are removed
#     - repeated slashes are squeezed except a leading doubled slash '//'
#       (which might indicate a networked disk on some OS).
#
#   REFERENCE_STRING is used to turn '/' into '\' and vice-versa: if
#   REFERENCE_STRING contains some backslashes, all slashes and backslashes
#   are turned into backslashes, otherwise they are all turned into slashes.
#
#   This makes processing of DOS filenames quite easier, because you can
#   turn a filename to the Unix notation, make your processing, and turn it
#   back to original notation.
#
#     filename='A:\FOO\\BAR\'
#     old_filename="$filename"
#     # Switch to the unix notation
#     AX_NORMALIZE_PATH([filename], ["/"])
#     # now we have $filename = 'A:/FOO/BAR' and we can process it as if
#     # it was a Unix path.  For instance let's say that you want
#     # to append '/subpath':
#     filename="$filename/subpath"
#     # finally switch back to the original notation
#     AX_NORMALIZE_PATH([filename], ["$old_filename"])
#     # now $filename equals to 'A:\FOO\BAR\subpath'
#
#   One good reason to make all path processing with the unix convention is
#   that backslashes have a special meaning in many cases. For instance
#
#     expr 'A:\FOO' : 'A:\Foo'
#
#   will return 0 because the second argument is a regex in which
#   backslashes have to be backslashed. In other words, to have the two
#   strings to match you should write this instead:
#
#     expr 'A:\Foo' : 'A:\\Foo'
#
#   Such behavior makes DOS filenames extremely unpleasant to work with. So
#   temporary turn your paths to the Unix notation, and revert them to the
#   original notation after the processing. See the macro
#   AX_COMPUTE_RELATIVE_PATHS for a concrete example of this.
#
#   REFERENCE_STRING defaults to $VARIABLE, this means that slashes will be
#   converted to backslashes if $VARIABLE already contains some backslashes
#   (see $thirddir below).
#
#     firstdir='/usr/local//share'
#     seconddir='C:\Program Files\\'
#     thirddir='C:\home/usr/'
#     AX_NORMALIZE_PATH([firstdir])
#     AX_NORMALIZE_PATH([seconddir])
#     AX_NORMALIZE_PATH([thirddir])
#     # $firstdir = '/usr/local/share'
#     # $seconddir = 'C:\Program Files'
#     # $thirddir = 'C:\home\usr'
#
# LICENSE
#
#   Copyright (c) 2008 Alexandre Duret-Lutz <adl@gnu.org>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <https://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

#serial 8

AU_ALIAS([ADL_NORMALIZE_PATH], [AX_NORMALIZE_PATH])
AC_DEFUN([AX_NORMALIZE_PATH],
[case ":[$]$1:" in
# change empty paths to '.'
  ::) $1='.' ;;
# strip trailing slashes
  :*[[\\/]]:) $1=`echo "[$]$1" | sed 's,[[\\/]]*[$],,'` ;;
  :*:) ;;
esac
# squeeze repeated slashes
case ifelse($2,,"[$]$1",$2) in
# if the path contains any backslashes, turn slashes into backslashes
 *\\*) $1=`echo "[$]$1" | sed 's,\(.\)[[\\/]][[\\/]]*,\1\\\\,g'` ;;
# if the path contains slashes, also turn backslashes into slashes
 *) $1=`echo "[$]$1" | sed 's,\(.\)[[\\/]][[\\/]]*,\1/,g'` ;;
esac])
