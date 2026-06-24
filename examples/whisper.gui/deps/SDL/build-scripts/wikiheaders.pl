#!/usr/bin/perl -w

use warnings;
use strict;
use Text::Wrap;

$Text::Wrap::huge = 'overflow';

my $projectfullname = 'Simple Directmedia Layer';
my $projectshortname = 'SDL';
my $wikisubdir = '';
my $incsubdir = 'include';
my $readmesubdir = undef;
my $apiprefixregex = undef;
my $versionfname = 'include/SDL_version.h';
my $versionmajorregex = '\A\#define\s+SDL_MAJOR_VERSION\s+(\d+)\Z';
my $versionminorregex = '\A\#define\s+SDL_MINOR_VERSION\s+(\d+)\Z';
my $versionpatchregex = '\A\#define\s+SDL_PATCHLEVEL\s+(\d+)\Z';
my $mainincludefname = 'SDL.h';
my $selectheaderregex = '\ASDL.*?\.h\Z';
my $projecturl = 'https://libsdl.org/';
my $wikiurl = 'https://wiki.libsdl.org';
my $bugreporturl = 'https://github.com/libsdl-org/sdlwiki/issues/new';
my $srcpath = undef;
my $wikipath = undef;
my $wikireadmesubdir = 'README';
my $warn_about_missing = 0;
my $copy_direction = 0;
my $optionsfname = undef;
my $wikipreamble = undef;
my $changeformat = undef;

foreach (@ARGV) {
    $warn_about_missing = 1, next if $_ eq '--warn-about-missing';
    $copy_direction = 1, next if $_ eq '--copy-to-headers';
    $copy_direction = 1, next if $_ eq '--copy-to-header';
    $copy_direction = -1, next if $_ eq '--copy-to-wiki';
    $copy_direction = -2, next if $_ eq '--copy-to-manpages';
    if (/\A--options=(.*)\Z/) {
        $optionsfname = $1;
        next;
    } elsif (/\A--changeformat=(.*)\Z/) {
        $changeformat = $1;
        next;
    }
    $srcpath = $_, next if not defined $srcpath;
    $wikipath = $_, next if not defined $wikipath;
}

my $default_optionsfname = '.wikiheaders-options';
$default_optionsfname = "$srcpath/$default_optionsfname" if defined $srcpath;

if ((not defined $optionsfname) && (-f $default_optionsfname)) {
    $optionsfname = $default_optionsfname;
}

if (defined $optionsfname) {
    open OPTIONS, '<', $optionsfname or die("Failed to open options file '$optionsfname': $!\n");
    while (<OPTIONS>) {
        chomp;
        if (/\A(.*?)\=(.*)\Z/) {
            my $key = $1;
            my $val = $2;
            $key =~ s/\A\s+//;
            $key =~ s/\s+\Z//;
            $val =~ s/\A\s+//;
            $val =~ s/\s+\Z//;
            $warn_about_missing = int($val), next if $key eq 'warn_about_missing';
            $srcpath = $val, next if $key eq 'srcpath';
            $wikipath = $val, next if $key eq 'wikipath';
            $apiprefixregex = $val, next if $key eq 'apiprefixregex';
            $projectfullname = $val, next if $key eq 'projectfullname';
            $projectshortname = $val, next if $key eq 'projectshortname';
            $wikisubdir = $val, next if $key eq 'wikisubdir';
            $incsubdir = $val, next if $key eq 'incsubdir';
            $readmesubdir = $val, next if $key eq 'readmesubdir';
            $versionmajorregex = $val, next if $key eq 'versionmajorregex';
            $versionminorregex = $val, next if $key eq 'versionminorregex';
            $versionpatchregex = $val, next if $key eq 'versionpatchregex';
            $versionfname = $val, next if $key eq 'versionfname';
            $mainincludefname = $val, next if $key eq 'mainincludefname';
            $selectheaderregex = $val, next if $key eq 'selectheaderregex';
            $projecturl = $val, next if $key eq 'projecturl';
            $wikiurl = $val, next if $key eq 'wikiurl';
            $bugreporturl = $val, next if $key eq 'bugreporturl';
            $wikipreamble = $val, next if $key eq 'wikipreamble';
        }
    }
    close(OPTIONS);
}

my $wordwrap_mode = 'mediawiki';
sub wordwrap_atom {   # don't call this directly.
    my $str = shift;
    my $retval = '';

    # wordwrap but leave links intact, even if they overflow.
    if ($wordwrap_mode eq 'mediawiki') {
        while ($str =~ s/(.*?)\s*(\[https?\:\/\/.*?\s+.*?\])\s*//ms) {
            $retval .= fill('', '', $1); # wrap it.
            $retval .= "\n$2\n";  # don't wrap it.
        }
    } elsif ($wordwrap_mode eq 'md') {
        while ($str =~ s/(.*?)\s*(\[.*?\]\(https?\:\/\/.*?\))\s*//ms) {
            $retval .= fill('', '', $1); # wrap it.
            $retval .= "\n$2\n";  # don't wrap it.
        }
    }

    return $retval . fill('', '', $str);
}

sub wordwrap_with_bullet_indent {  # don't call this directly.
    my $bullet = shift;
    my $str = shift;
    my $retval = '';

    #print("WORDWRAP BULLET ('$bullet'):\n\n$str\n\n");

    # You _can't_ (at least with Pandoc) have a bullet item with a newline in
    #  MediaWiki, so _remove_ wrapping!
    if ($wordwrap_mode eq 'mediawiki') {
        $retval = "$bullet$str";
        $retval =~ s/\n/ /gms;
        $retval =~ s/\s+$//gms;
        #print("WORDWRAP BULLET DONE:\n\n$retval\n\n");
        return "$retval\n";
    }

    my $bulletlen = length($bullet);

    # wrap it and then indent each line to be under the bullet.
    $Text::Wrap::columns -= $bulletlen;
    my @wrappedlines = split /\n/, wordwrap_atom($str);
    $Text::Wrap::columns += $bulletlen;

    my $prefix = $bullet;
    my $usual_prefix = ' ' x $bulletlen;

    foreach (@wrappedlines) {
        s/\s*\Z//;
        $retval .= "$prefix$_\n";
        $prefix = $usual_prefix;
    }

    return $retval;
}

sub wordwrap_one_paragraph {  # don't call this directly.
    my $retval = '';
    my $p = shift;
    #print "\n\n\nPARAGRAPH: [$p]\n\n\n";
    if ($p =~ s/\A([\*\-] )//) {  # bullet list, starts with "* " or "- ".
        my $bullet = $1;
        my $item = '';
        my @items = split /\n/, $p;
        foreach (@items) {
            if (s/\A([\*\-] )//) {
                $retval .= wordwrap_with_bullet_indent($bullet, $item);
                $item = '';
            }
            s/\A\s*//;
            $item .= "$_\n";   # accumulate lines until we hit the end or another bullet.
        }
        if ($item ne '') {
            $retval .= wordwrap_with_bullet_indent($bullet, $item);
        }
    } else {
        $retval = wordwrap_atom($p) . "\n";
    }

    return $retval;
}

sub wordwrap_paragraphs {  # don't call this directly.
    my $str = shift;
    my $retval = '';
    my @paragraphs = split /\n\n/, $str;
    foreach (@paragraphs) {
        next if $_ eq '';
        $retval .= wordwrap_one_paragraph($_);
        $retval .= "\n";
    }
    return $retval;
}

my $wordwrap_default_columns = 76;
sub wordwrap {
    my $str = shift;
    my $columns = shift;

    $columns = $wordwrap_default_columns if not defined $columns;
    $columns += $wordwrap_default_columns if $columns < 0;
    $Text::Wrap::columns = $columns;

    my $retval = '';

    #print("\n\nWORDWRAP:\n\n$str\n\n\n");

    $str =~ s/\A\n+//ms;

    while ($str =~ s/(.*?)(\`\`\`.*?\`\`\`|\<syntaxhighlight.*?\<\/syntaxhighlight\>)//ms) {
        #print("\n\nWORDWRAP BLOCK:\n\n$1\n\n ===\n\n$2\n\n\n");
        $retval .= wordwrap_paragraphs($1); # wrap it.
        $retval .= "$2\n\n";  # don't wrap it.
    }

    $retval .= wordwrap_paragraphs($str);  # wrap what's left.
    $retval =~ s/\n+\Z//ms;

    #print("\n\nWORDWRAP DONE:\n\n$retval\n\n\n");
    return $retval;
}

# This assumes you're moving from Markdown (in the Doxygen data) to Wiki, which
#  is why the 'md' section is so sparse.
sub wikify_chunk {
    my $wikitype = shift;
    my $str = shift;
    my $codelang = shift;
    my $code = shift;

    #print("\n\nWIKIFY CHUNK:\n\n$str\n\n\n");

    if ($wikitype eq 'mediawiki') {
        # convert `code` things first, so they aren't mistaken for other markdown items.
        my $codedstr = '';
        while ($str =~ s/\A(.*?)\`(.*?)\`//ms) {
            my $codeblock = $2;
            $codedstr .= wikify_chunk($wikitype, $1, undef, undef);
            if (defined $apiprefixregex) {
                # Convert obvious API things to wikilinks, even inside `code` blocks.
                $codeblock =~ s/\b($apiprefixregex[a-zA-Z0-9_]+)/[[$1]]/gms;
            }
            $codedstr .= "<code>$codeblock</code>";
        }

        # Convert obvious API things to wikilinks.
        if (defined $apiprefixregex) {
            $str =~ s/\b($apiprefixregex[a-zA-Z0-9_]+)/[[$1]]/gms;
        }

        # Make some Markdown things into MediaWiki...

        # links
        $str =~ s/\[(.*?)\]\((https?\:\/\/.*?)\)/\[$2 $1\]/g;

        # bold+italic
        $str =~ s/\*\*\*(.*?)\*\*\*/'''''$1'''''/gms;

        # bold
        $str =~ s/\*\*(.*?)\*\*/'''$1'''/gms;

        # italic
        $str =~ s/\*(.*?)\*/''$1''/gms;

        # bullets
        $str =~ s/^\- /* /gm;

        $str = $codedstr . $str;

        if (defined $code) {
            $str .= "<syntaxhighlight lang='$codelang'>$code<\/syntaxhighlight>";
        }
    } elsif ($wikitype eq 'md') {
        # convert `code` things first, so they aren't mistaken for other markdown items.
        my $codedstr = '';
        while ($str =~ s/\A(.*?)(\`.*?\`)//ms) {
            my $codeblock = $2;
            $codedstr .= wikify_chunk($wikitype, $1, undef, undef);
            if (defined $apiprefixregex) {
                # Convert obvious API things to wikilinks, even inside `code` blocks,
                # BUT ONLY IF the entire code block is the API thing,
                # So something like "just call `SDL_Whatever`" will become
                # "just call [`SDL_Whatever`](SDL_Whatever)", but
                # "just call `SDL_Whatever(7)`" will not. It's just the safest
                # way to do this without resorting to wrapping things in html <code> tags.
                $codeblock =~ s/\A\`($apiprefixregex[a-zA-Z0-9_]+)\`\Z/[`$1`]($1)/gms;
            }
            $codedstr .= $codeblock;
        }

        # Convert obvious API things to wikilinks.
        if (defined $apiprefixregex) {
            $str =~ s/\b($apiprefixregex[a-zA-Z0-9_]+)/[$1]($1)/gms;
        }

        $str = $codedstr . $str;

        if (defined $code) {
            $str .= "```$codelang$code```";
        }
    }

    #print("\n\nWIKIFY CHUNK DONE:\n\n$str\n\n\n");

    return $str;
}

sub wikify {
    my $wikitype = shift;
    my $str = shift;
    my $retval = '';

    #print("WIKIFY WHOLE:\n\n$str\n\n\n");

    while ($str =~ s/\A(.*?)\`\`\`(c\+\+|c)(.*?)\`\`\`//ms) {
        $retval .= wikify_chunk($wikitype, $1, $2, $3);
    }
    $retval .= wikify_chunk($wikitype, $str, undef, undef);

    #print("WIKIFY WHOLE DONE:\n\n$retval\n\n\n");

    return $retval;
}


my $dewikify_mode = 'md';
my $dewikify_manpage_code_indent = 1;

sub dewikify_chunk {
    my $wikitype = shift;
    my $str = shift;
    my $codelang = shift;
    my $code = shift;

    #print("\n\nDEWIKIFY CHUNK:\n\n$str\n\n\n");

    if ($dewikify_mode eq 'md') {
        if ($wikitype eq 'mediawiki') {
            # Doxygen supports Markdown (and it just simply looks better than MediaWiki
            # when looking at the raw headers), so do some conversions here as necessary.

            # Dump obvious wikilinks.
            if (defined $apiprefixregex) {
                $str =~ s/\[\[($apiprefixregex[a-zA-Z0-9_]+)\]\]/$1/gms;
            }

            # links
            $str =~ s/\[(https?\:\/\/.*?)\s+(.*?)\]/\[$2\]\($1\)/g;

            # <code></code> is also popular.  :/
            $str =~ s/\<code>(.*?)<\/code>/`$1`/gms;

            # bold+italic
            $str =~ s/'''''(.*?)'''''/***$1***/gms;

            # bold
            $str =~ s/'''(.*?)'''/**$1**/gms;

            # italic
            $str =~ s/''(.*?)''/*$1*/gms;

            # bullets
            $str =~ s/^\* /- /gm;
        } elsif ($wikitype eq 'md') {
            # Dump obvious wikilinks. The rest can just passthrough.
            if (defined $apiprefixregex) {
                $str =~ s/\[(\`?$apiprefixregex[a-zA-Z0-9_]+\`?)\]\($apiprefixregex[a-zA-Z0-9_]+\)/$1/gms;
            }
        }

        if (defined $code) {
            $str .= "```$codelang$code```";
        }
    } elsif ($dewikify_mode eq 'manpage') {
        $str =~ s/\./\\[char46]/gms;  # make sure these can't become control codes.
        if ($wikitype eq 'mediawiki') {
            # Dump obvious wikilinks.
            if (defined $apiprefixregex) {
                $str =~ s/\s*\[\[($apiprefixregex[a-zA-Z0-9_]+)\]\]\s*/\n.BR $1\n/gms;
            }

            # links
            $str =~ s/\[(https?\:\/\/.*?)\s+(.*?)\]/\n.URL "$1" "$2"\n/g;

            # <code></code> is also popular.  :/
            $str =~ s/\s*\<code>(.*?)<\/code>\s*/\n.BR $1\n/gms;

            # bold+italic
            $str =~ s/\s*'''''(.*?)'''''\s*/\n.BI $1\n/gms;

            # bold
            $str =~ s/\s*'''(.*?)'''\s*/\n.B $1\n/gms;

            # italic
            $str =~ s/\s*''(.*?)''\s*/\n.I $1\n/gms;

            # bullets
            $str =~ s/^\* /\n\\\(bu /gm;
        } elsif ($wikitype eq 'md') {
            # Dump obvious wikilinks.
            if (defined $apiprefixregex) {
                $str =~ s/\[(\`?$apiprefixregex[a-zA-Z0-9_]+\`?)\]\($apiprefixregex[a-zA-Z0-9_]+\)/\n.BR $1\n/gms;
            }

            # links
            $str =~ s/\[(.*?)]\((https?\:\/\/.*?)\)/\n.URL "$2" "$1"\n/g;

            # <code></code> is also popular.  :/
            $str =~ s/\s*\`(.*?)\`\s*/\n.BR $1\n/gms;

            # bold+italic
            $str =~ s/\s*\*\*\*(.*?)\*\*\*\s*/\n.BI $1\n/gms;

            # bold
            $str =~ s/\s*\*\*(.*?)\*\*\s*/\n.B $1\n/gms;

            # italic
            $str =~ s/\s*\*(.*?)\*\s*/\n.I $1\n/gms;

            # bullets
            $str =~ s/^\- /\n\\\(bu /gm;

        } else {
            die("Unexpected wikitype when converting to manpages\n");   # !!! FIXME: need to handle Markdown wiki pages.
        }

        if (defined $code) {
            $code =~ s/\A\n+//gms;
            $code =~ s/\n+\Z//gms;
            if ($dewikify_manpage_code_indent) {
                $str .= "\n.IP\n"
            } else {
                $str .= "\n.PP\n"
            }
            $str .= ".EX\n$code\n.EE\n.PP\n";
        }
    } else {
        die("Unexpected dewikify_mode\n");
    }

    #print("\n\nDEWIKIFY CHUNK DONE:\n\n$str\n\n\n");

    return $str;
}

sub dewikify {
    my $wikitype = shift;
    my $str = shift;
    return '' if not defined $str;

    #print("DEWIKIFY WHOLE:\n\n$str\n\n\n");

    $str =~ s/\A[\s\n]*\= .*? \=\s*?\n+//ms;
    $str =~ s/\A[\s\n]*\=\= .*? \=\=\s*?\n+//ms;

    my $retval = '';
    while ($str =~ s/\A(.*?)<syntaxhighlight lang='?(.*?)'?>(.*?)<\/syntaxhighlight\>//ms) {
        $retval .= dewikify_chunk($wikitype, $1, $2, $3);
    }
    $retval .= dewikify_chunk($wikitype, $str, undef, undef);

    #print("DEWIKIFY WHOLE DONE:\n\n$retval\n\n\n");

    return $retval;
}

sub filecopy {
    my $src = shift;
    my $dst = shift;
    my $endline = shift;
    $endline = "\n" if not defined $endline;

    open(COPYIN, '<', $src) or die("Failed to open '$src' for reading: $!\n");
    open(COPYOUT, '>', $dst) or die("Failed to open '$dst' for writing: $!\n");
    while (<COPYIN>) {
        chomp;
        s/[ \t\r\n]*\Z//;
        print COPYOUT "$_$endline";
    }
    close(COPYOUT);
    close(COPYIN);
}

sub usage {
    die("USAGE: $0 <source code git clone path> <wiki git clone path> [--copy-to-headers|--copy-to-wiki|--copy-to-manpages] [--warn-about-missing]\n\n");
}

usage() if not defined $srcpath;
usage() if not defined $wikipath;
#usage() if $copy_direction == 0;

my @standard_wiki_sections = (
    'Draft',
    '[Brief]',
    'Deprecated',
    'Syntax',
    'Function Parameters',
    'Return Value',
    'Remarks',
    'Thread Safety',
    'Version',
    'Code Examples',
    'Related Functions'
);

# Sections that only ever exist in the wiki and shouldn't be deleted when
#  not found in the headers.
my %only_wiki_sections = (  # The ones don't mean anything, I just need to check for key existence.
    'Draft', 1,
    'Code Examples', 1
);


my %headers = ();       # $headers{"SDL_audio.h"} -> reference to an array of all lines of text in SDL_audio.h.
my %headerfuncs = ();   # $headerfuncs{"SDL_OpenAudio"} -> string of header documentation for SDL_OpenAudio, with comment '*' bits stripped from the start. Newlines embedded!
my %headerdecls = ();
my %headerfuncslocation = ();   # $headerfuncslocation{"SDL_OpenAudio"} -> name of header holding SDL_OpenAudio define ("SDL_audio.h" in this case).
my %headerfuncschunk = ();   # $headerfuncschunk{"SDL_OpenAudio"} -> offset in array in %headers that should be replaced for this function.
my %headerfuncshasdoxygen = ();   # $headerfuncschunk{"SDL_OpenAudio"} -> 1 if there was no existing doxygen for this function.

my $incpath = "$srcpath";
$incpath .= "/$incsubdir" if $incsubdir ne '';

my $wikireadmepath = "$wikipath/$wikireadmesubdir";
my $readmepath = undef;
if (defined $readmesubdir) {
    $readmepath = "$srcpath/$readmesubdir";
}

opendir(DH, $incpath) or die("Can't opendir '$incpath': $!\n");
while (my $d = readdir(DH)) {
    my $dent = $d;
    next if not $dent =~ /$selectheaderregex/;  # just selected headers.
    open(FH, '<', "$incpath/$dent") or die("Can't open '$incpath/$dent': $!\n");

    my @contents = ();

    while (<FH>) {
        chomp;
        my $decl;
        my @templines;
        my $str;
        my $has_doxygen = 1;
        if (/\A\s*extern\s+(SDL_DEPRECATED\s+|)DECLSPEC/) {  # a function declaration without a doxygen comment?
            @templines = ();
            $decl = $_;
            $str = '';
            $has_doxygen = 0;
        } elsif (not /\A\/\*\*\s*\Z/) {  # not doxygen comment start?
            push @contents, $_;
            next;
        } else {   # Start of a doxygen comment, parse it out.
            @templines = ( $_ );
            while (<FH>) {
                chomp;
                push @templines, $_;
                last if /\A\s*\*\/\Z/;
                if (s/\A\s*\*\s*\`\`\`/```/) {  # this is a hack, but a lot of other code relies on the whitespace being trimmed, but we can't trim it in code blocks...
                    $str .= "$_\n";
                    while (<FH>) {
                        chomp;
                        push @templines, $_;
                        s/\A\s*\*\s?//;
                        if (s/\A\s*\`\`\`/```/) {
                            $str .= "$_\n";
                            last;
                        } else {
                            $str .= "$_\n";
                        }
                    }
                } else {
                    s/\A\s*\*\s*//;
                    $str .= "$_\n";
                }
            }

            $decl = <FH>;
            $decl = '' if not defined $decl;
            chomp($decl);
            if (not $decl =~ /\A\s*extern\s+(SDL_DEPRECATED\s+|)DECLSPEC/) {
                #print "Found doxygen but no function sig:\n$str\n\n";
                foreach (@templines) {
                    push @contents, $_;
                }
                push @contents, $decl;
                next;
            }
        }

        my @decllines = ( $decl );

        if (not $decl =~ /\)\s*;/) {
            while (<FH>) {
                chomp;
                push @decllines, $_;
                s/\A\s+//;
                s/\s+\Z//;
                $decl .= " $_";
                last if /\)\s*;/;
            }
        }

        $decl =~ s/\s+\);\Z/);/;
        $decl =~ s/\s+\Z//;
        #print("DECL: [$decl]\n");

        my $fn = '';
        if ($decl =~ /\A\s*extern\s+(SDL_DEPRECATED\s+|)DECLSPEC\s+(const\s+|)(unsigned\s+|)(.*?)\s*(\*?)\s*SDLCALL\s+(.*?)\s*\((.*?)\);/) {
            $fn = $6;
            #$decl =~ s/\A\s*extern\s+DECLSPEC\s+(.*?)\s+SDLCALL/$1/;
        } else {
            #print "Found doxygen but no function sig:\n$str\n\n";
            foreach (@templines) {
                push @contents, $_;
            }
            foreach (@decllines) {
                push @contents, $_;
            }
            next;
        }

        $decl = '';  # build this with the line breaks, since it looks better for syntax highlighting.
        foreach (@decllines) {
            if ($decl eq '') {
                $decl = $_;
                $decl =~ s/\Aextern\s+(SDL_DEPRECATED\s+|)DECLSPEC\s+(.*?)\s+(\*?)SDLCALL\s+/$2$3 /;
            } else {
                my $trimmed = $_;
                # !!! FIXME: trim space for SDL_DEPRECATED if it was used, too.
                $trimmed =~ s/\A\s{24}//;  # 24 for shrinking to match the removed "extern DECLSPEC SDLCALL "
                $decl .= $trimmed;
            }
            $decl .= "\n";
        }

        #print("$fn:\n$str\n\n");

        # There might be multiple declarations of a function due to #ifdefs,
        #  and only one of them will have documentation. If we hit an
        #  undocumented one before, delete the placeholder line we left for
        #  it so it doesn't accumulate a new blank line on each run.
        my $skipfn = 0;
        if (defined $headerfuncshasdoxygen{$fn}) {
            if ($headerfuncshasdoxygen{$fn} == 0) {  # An undocumented declaration already exists, nuke its placeholder line.
                delete $contents[$headerfuncschunk{$fn}];  # delete DOES NOT RENUMBER existing elements!
            } else {  # documented function already existed?
                $skipfn = 1;  # don't add this copy to the list of functions.
                if ($has_doxygen) {
                    print STDERR "WARNING: Function '$fn' appears to be documented in multiple locations. Only keeping the first one we saw!\n";
                }
                push @contents, join("\n", @decllines);  # just put the existing declation in as-is.
            }
        }

        if (!$skipfn) {
            $headerfuncs{$fn} = $str;
            $headerdecls{$fn} = $decl;
            $headerfuncslocation{$fn} = $dent;
            $headerfuncschunk{$fn} = scalar(@contents);
            $headerfuncshasdoxygen{$fn} = $has_doxygen;
            push @contents, join("\n", @templines);
            push @contents, join("\n", @decllines);
        }

    }
    close(FH);

    $headers{$dent} = \@contents;
}
closedir(DH);


# !!! FIXME: we need to parse enums and typedefs and structs and defines and and and and and...
# !!! FIXME:  (but functions are good enough for now.)

my %wikitypes = ();  # contains string of wiki page extension, like $wikitypes{"SDL_OpenAudio"} == 'mediawiki'
my %wikifuncs = ();  # contains references to hash of strings, each string being the full contents of a section of a wiki page, like $wikifuncs{"SDL_OpenAudio"}{"Remarks"}.
my %wikisectionorder = ();   # contains references to array, each array item being a key to a wikipage section in the correct order, like $wikisectionorder{"SDL_OpenAudio"}[2] == 'Remarks'
opendir(DH, $wikipath) or die("Can't opendir '$wikipath': $!\n");
while (my $d = readdir(DH)) {
    my $dent = $d;
    my $type = '';
    if ($dent =~ /\.(md|mediawiki)\Z/) {
        $type = $1;
    } else {
        next;  # only dealing with wiki pages.
    }

    my $fn = $dent;
    $fn =~ s/\..*\Z//;

    # Ignore FrontPage.
    next if $fn eq 'FrontPage';

    # Ignore "Category*" pages.
    next if ($fn =~ /\ACategory/);

    open(FH, '<', "$wikipath/$dent") or die("Can't open '$wikipath/$dent': $!\n");

    my $current_section = '[start]';
    my @section_order = ( $current_section );
    my %sections = ();
    $sections{$current_section} = '';

    my $firstline = 1;

    while (<FH>) {
        chomp;
        my $orig = $_;
        s/\A\s*//;
        s/\s*\Z//;

        if ($type eq 'mediawiki') {
            if (defined($wikipreamble) && $firstline && /\A\=\=\=\=\=\= (.*?) \=\=\=\=\=\=\Z/ && ($1 eq $wikipreamble)) {
                $firstline = 0;  # skip this.
                next;
            } elsif (/\A\= (.*?) \=\Z/) {
                $firstline = 0;
                $current_section = ($1 eq $fn) ? '[Brief]' : $1;
                die("Doubly-defined section '$current_section' in '$dent'!\n") if defined $sections{$current_section};
                push @section_order, $current_section;
                $sections{$current_section} = '';
            } elsif (/\A\=\= (.*?) \=\=\Z/) {
                $firstline = 0;
                $current_section = ($1 eq $fn) ? '[Brief]' : $1;
                die("Doubly-defined section '$current_section' in '$dent'!\n") if defined $sections{$current_section};
                push @section_order, $current_section;
                $sections{$current_section} = '';
                next;
            } elsif (/\A\-\-\-\-\Z/) {
                $firstline = 0;
                $current_section = '[footer]';
                die("Doubly-defined section '$current_section' in '$dent'!\n") if defined $sections{$current_section};
                push @section_order, $current_section;
                $sections{$current_section} = '';
                next;
            }
        } elsif ($type eq 'md') {
            if (defined($wikipreamble) && $firstline && /\A\#\#\#\#\#\# (.*?)\Z/ && ($1 eq $wikipreamble)) {
                $firstline = 0;  # skip this.
                next;
            } elsif (/\A\#+ (.*?)\Z/) {
                $firstline = 0;
                $current_section = ($1 eq $fn) ? '[Brief]' : $1;
                die("Doubly-defined section '$current_section' in '$dent'!\n") if defined $sections{$current_section};
                push @section_order, $current_section;
                $sections{$current_section} = '';
                next;
            } elsif (/\A\-\-\-\-\Z/) {
                $firstline = 0;
                $current_section = '[footer]';
                die("Doubly-defined section '$current_section' in '$dent'!\n") if defined $sections{$current_section};
                push @section_order, $current_section;
                $sections{$current_section} = '';
                next;
            }
        } else {
            die("Unexpected wiki file type. Fixme!\n");
        }

        if ($firstline) {
            $firstline = ($_ ne '');
        }
        if (!$firstline) {
            $sections{$current_section} .= "$orig\n";
        }
    }
    close(FH);

    foreach (keys %sections) {
        $sections{$_} =~ s/\A\n+//;
        $sections{$_} =~ s/\n+\Z//;
        $sections{$_} .= "\n";
    }

    if (0) {
        foreach (@section_order) {
            print("$fn SECTION '$_':\n");
            print($sections{$_});
            print("\n\n");
        }
    }

    $wikitypes{$fn} = $type;
    $wikifuncs{$fn} = \%sections;
    $wikisectionorder{$fn} = \@section_order;
}
closedir(DH);


if ($warn_about_missing) {
    foreach (keys %wikifuncs) {
        my $fn = $_;
        if (not defined $headerfuncs{$fn}) {
            print("WARNING: $fn defined in the wiki but not the headers!\n");
        }
    }

    foreach (keys %headerfuncs) {
        my $fn = $_;
        if (not defined $wikifuncs{$fn}) {
            print("WARNING: $fn defined in the headers but not the wiki!\n");
        }
    }
}

if ($copy_direction == 1) {  # --copy-to-headers
    my %changed_headers = ();

    $dewikify_mode = 'md';
    $wordwrap_mode = 'md';   # the headers use Markdown format.

    foreach (keys %headerfuncs) {
        my $fn = $_;
        next if not defined $wikifuncs{$fn};  # don't have a page for that function, skip it.
        my $wikitype = $wikitypes{$fn};
        my $sectionsref = $wikifuncs{$fn};
        my $remarks = $sectionsref->{'Remarks'};
        my $params = $sectionsref->{'Function Parameters'};
        my $returns = $sectionsref->{'Return Value'};
        my $threadsafety = $sectionsref->{'Thread Safety'};
        my $version = $sectionsref->{'Version'};
        my $related = $sectionsref->{'Related Functions'};
        my $deprecated = $sectionsref->{'Deprecated'};
        my $brief = $sectionsref->{'[Brief]'};
        my $addblank = 0;
        my $str = '';

        $headerfuncshasdoxygen{$fn} = 1;  # Added/changed doxygen for this header.

        $brief = dewikify($wikitype, $brief);
        $brief =~ s/\A(.*?\.) /$1\n/;  # \brief should only be one sentence, delimited by a period+space. Split if necessary.
        my @briefsplit = split /\n/, $brief;
        $brief = shift @briefsplit;

        if (defined $remarks) {
            $remarks = join("\n", @briefsplit) . dewikify($wikitype, $remarks);
        }

        if (defined $brief) {
            $str .= "\n" if $addblank; $addblank = 1;
            $str .= wordwrap($brief) . "\n";
        }

        if (defined $remarks) {
            $str .= "\n" if $addblank; $addblank = 1;
            $str .= wordwrap($remarks) . "\n";
        }

        if (defined $deprecated) {
            # !!! FIXME: lots of code duplication in all of these.
            $str .= "\n" if $addblank; $addblank = 1;
            my $v = dewikify($wikitype, $deprecated);
            my $whitespacelen = length("\\deprecated") + 1;
            my $whitespace = ' ' x $whitespacelen;
            $v = wordwrap($v, -$whitespacelen);
            my @desclines = split /\n/, $v;
            my $firstline = shift @desclines;
            $str .= "\\deprecated $firstline\n";
            foreach (@desclines) {
                $str .= "${whitespace}$_\n";
            }
        }

        if (defined $params) {
            $str .= "\n" if $addblank; $addblank = (defined $returns) ? 0 : 1;
            my @lines = split /\n/, dewikify($wikitype, $params);
            if ($wikitype eq 'mediawiki') {
                die("Unexpected data parsing MediaWiki table") if (shift @lines ne '{|');  # Dump the '{|' start
                while (scalar(@lines) >= 3) {
                    my $name = shift @lines;
                    my $desc = shift @lines;
                    my $terminator = shift @lines;  # the '|-' or '|}' line.
                    last if ($terminator ne '|-') and ($terminator ne '|}');  # we seem to have run out of table.
                    $name =~ s/\A\|\s*//;
                    $name =~ s/\A\*\*(.*?)\*\*/$1/;
                    $name =~ s/\A\'\'\'(.*?)\'\'\'/$1/;
                    $desc =~ s/\A\|\s*//;
                    #print STDERR "FN: $fn   NAME: $name   DESC: $desc TERM: $terminator\n";
                    my $whitespacelen = length($name) + 8;
                    my $whitespace = ' ' x $whitespacelen;
                    $desc = wordwrap($desc, -$whitespacelen);
                    my @desclines = split /\n/, $desc;
                    my $firstline = shift @desclines;
                    $str .= "\\param $name $firstline\n";
                    foreach (@desclines) {
                        $str .= "${whitespace}$_\n";
                    }
                }
            } elsif ($wikitype eq 'md') {
                my $l;
                $l = shift @lines;
                die("Unexpected data parsing Markdown table") if (not $l =~ /\A\s*\|\s*\|\s*\|\s*\Z/);
                $l = shift @lines;
                die("Unexpected data parsing Markdown table") if (not $l =~ /\A\s*\|\s*\-*\s*\|\s*\-*\s*\|\s*\Z/);
                while (scalar(@lines) >= 1) {
                    $l = shift @lines;
                    if ($l =~ /\A\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*\Z/) {
                        my $name = $1;
                        my $desc = $2;
                        $name =~ s/\A\*\*(.*?)\*\*/$1/;
                        $name =~ s/\A\'\'\'(.*?)\'\'\'/$1/;
                        #print STDERR "FN: $fn   NAME: $name   DESC: $desc\n";
                        my $whitespacelen = length($name) + 8;
                        my $whitespace = ' ' x $whitespacelen;
                        $desc = wordwrap($desc, -$whitespacelen);
                        my @desclines = split /\n/, $desc;
                        my $firstline = shift @desclines;
                        $str .= "\\param $name $firstline\n";
                        foreach (@desclines) {
                            $str .= "${whitespace}$_\n";
                        }
                    } else {
                        last;  # we seem to have run out of table.
                    }
                }
            } else {
                die("write me");
            }
        }

        if (defined $returns) {
            $str .= "\n" if $addblank; $addblank = 1;
            my $r = dewikify($wikitype, $returns);
            my $retstr = "\\returns";
            if ($r =~ s/\AReturn(s?) //) {
                $retstr = "\\return$1";
            }

            my $whitespacelen = length($retstr) + 1;
            my $whitespace = ' ' x $whitespacelen;
            $r = wordwrap($r, -$whitespacelen);
            my @desclines = split /\n/, $r;
            my $firstline = shift @desclines;
            $str .= "$retstr $firstline\n";
            foreach (@desclines) {
                $str .= "${whitespace}$_\n";
            }
        }

        if (defined $threadsafety) {
            # !!! FIXME: lots of code duplication in all of these.
            $str .= "\n" if $addblank; $addblank = 1;
            my $v = dewikify($wikitype, $threadsafety);
            my $whitespacelen = length("\\threadsafety") + 1;
            my $whitespace = ' ' x $whitespacelen;
            $v = wordwrap($v, -$whitespacelen);
            my @desclines = split /\n/, $v;
            my $firstline = shift @desclines;
            $str .= "\\threadsafety $firstline\n";
            foreach (@desclines) {
                $str .= "${whitespace}$_\n";
            }
        }

        if (defined $version) {
            # !!! FIXME: lots of code duplication in all of these.
            $str .= "\n" if $addblank; $addblank = 1;
            my $v = dewikify($wikitype, $version);
            my $whitespacelen = length("\\since") + 1;
            my $whitespace = ' ' x $whitespacelen;
            $v = wordwrap($v, -$whitespacelen);
            my @desclines = split /\n/, $v;
            my $firstline = shift @desclines;
            $str .= "\\since $firstline\n";
            foreach (@desclines) {
                $str .= "${whitespace}$_\n";
            }
        }

        if (defined $related) {
            # !!! FIXME: lots of code duplication in all of these.
            $str .= "\n" if $addblank; $addblank = 1;
            my $v = dewikify($wikitype, $related);
            my @desclines = split /\n/, $v;
            foreach (@desclines) {
                s/\A(\:|\* )//;
                s/\(\)\Z//;  # Convert "SDL_Func()" to "SDL_Func"
                s/\[\[(.*?)\]\]/$1/;  # in case some wikilinks remain.
                s/\[(.*?)\]\(.*?\)/$1/;  # in case some wikilinks remain.
                s/\A\/*//;
                $str .= "\\sa $_\n";
            }
        }

        my $header = $headerfuncslocation{$fn};
        my $contentsref = $headers{$header};
        my $chunk = $headerfuncschunk{$fn};

        my @lines = split /\n/, $str;

        my $addnewline = (($chunk > 0) && ($$contentsref[$chunk-1] ne '')) ? "\n" : '';

        my $output = "$addnewline/**\n";
        foreach (@lines) {
            chomp;
            s/\s*\Z//;
            if ($_ eq '') {
                $output .= " *\n";
            } else {
                $output .= " * $_\n";
            }
        }
        $output .= " */";

        #print("$fn:\n$output\n\n");

        $$contentsref[$chunk] = $output;
        #$$contentsref[$chunk+1] = $headerdecls{$fn};

        $changed_headers{$header} = 1;
    }

    foreach (keys %changed_headers) {
        my $header = $_;

        # this is kinda inefficient, but oh well.
        my @removelines = ();
        foreach (keys %headerfuncslocation) {
            my $fn = $_;
            next if $headerfuncshasdoxygen{$fn};
            next if $headerfuncslocation{$fn} ne $header;
            # the index of the blank line we put before the function declaration in case we needed to replace it with new content from the wiki.
            push @removelines, $headerfuncschunk{$fn};
        }

        my $contentsref = $headers{$header};
        foreach (@removelines) {
            delete $$contentsref[$_];  # delete DOES NOT RENUMBER existing elements!
        }

        my $path = "$incpath/$header.tmp";
        open(FH, '>', $path) or die("Can't open '$path': $!\n");
        foreach (@$contentsref) {
            print FH "$_\n" if defined $_;
        }
        close(FH);
        rename($path, "$incpath/$header") or die("Can't rename '$path' to '$incpath/$header': $!\n");
    }

    if (defined $readmepath) {
        if ( -d $wikireadmepath ) {
            mkdir($readmepath);  # just in case
            opendir(DH, $wikireadmepath) or die("Can't opendir '$wikireadmepath': $!\n");
            while (readdir(DH)) {
                my $dent = $_;
                if ($dent =~ /\A(.*?)\.md\Z/) {  # we only bridge Markdown files here.
                    next if $1 eq 'FrontPage';
                    filecopy("$wikireadmepath/$dent", "$readmepath/README-$dent", "\r\n");
                }
            }
            closedir(DH);
        }
    }
} elsif ($copy_direction == -1) { # --copy-to-wiki

    if (defined $changeformat) {
        $dewikify_mode = $changeformat;
        $wordwrap_mode = $changeformat;
    }

    foreach (keys %headerfuncs) {
        my $fn = $_;
        next if not $headerfuncshasdoxygen{$fn};
        my $origwikitype = defined $wikitypes{$fn} ? $wikitypes{$fn} : 'md';  # default to MarkDown for new stuff.
        my $wikitype = (defined $changeformat) ? $changeformat : $origwikitype;
        die("Unexpected wikitype '$wikitype'\n") if (($wikitype ne 'mediawiki') and ($wikitype ne 'md') and ($wikitype ne 'manpage'));

        #print("$fn\n"); next;

        $wordwrap_mode = $wikitype;

        my $raw = $headerfuncs{$fn};  # raw doxygen text with comment characters stripped from start/end and start of each line.
        next if not defined $raw;
        $raw =~ s/\A\s*\\brief\s+//;  # Technically we don't need \brief (please turn on JAVADOC_AUTOBRIEF if you use Doxygen), so just in case one is present, strip it.

        my @doxygenlines = split /\n/, $raw;
        my $brief = '';
        while (@doxygenlines) {
            last if $doxygenlines[0] =~ /\A\\/;  # some sort of doxygen command, assume we're past the general remarks.
            last if $doxygenlines[0] =~ /\A\s*\Z/;  # blank line? End of paragraph, done.
            my $l = shift @doxygenlines;
            chomp($l);
            $l =~ s/\A\s*//;
            $l =~ s/\s*\Z//;
            $brief .= "$l ";
        }

        $brief =~ s/\A(.*?\.) /$1\n\n/;  # \brief should only be one sentence, delimited by a period+space. Split if necessary.
        my @briefsplit = split /\n/, $brief;
        $brief = wikify($wikitype, shift @briefsplit) . "\n";
        @doxygenlines = (@briefsplit, @doxygenlines);

        my $remarks = '';
        # !!! FIXME: wordwrap and wikify might handle this, now.
        while (@doxygenlines) {
            last if $doxygenlines[0] =~ /\A\\/;  # some sort of doxygen command, assume we're past the general remarks.
            my $l = shift @doxygenlines;
            if ($l =~ /\A\`\`\`/) {  # syntax highlighting, don't reformat.
                $remarks .= "$l\n";
                while ((@doxygenlines) && (not $l =~ /\`\`\`\Z/)) {
                    $l = shift @doxygenlines;
                    $remarks .= "$l\n";
                }
            } else {
                $l =~ s/\A\s*//;
                $l =~ s/\s*\Z//;
                $remarks .= "$l\n";
            }
        }

        #print("REMARKS:\n\n $remarks\n\n");

        $remarks = wordwrap(wikify($wikitype, $remarks));
        $remarks =~ s/\A\s*//;
        $remarks =~ s/\s*\Z//;

        my $decl = $headerdecls{$fn};
        #$decl =~ s/\*\s+SDLCALL/ *SDLCALL/;  # Try to make "void * Function" become "void *Function"
        #$decl =~ s/\A\s*extern\s+(SDL_DEPRECATED\s+|)DECLSPEC\s+(.*?)\s+(\*?)SDLCALL/$2$3/;

        my $syntax = '';
        if ($wikitype eq 'mediawiki') {
            $syntax = "<syntaxhighlight lang='c'>\n$decl</syntaxhighlight>\n";
        } elsif ($wikitype eq 'md') {
            $syntax = "```c\n$decl\n```\n";
        } else { die("Expected wikitype '$wikitype'\n"); }

        my %sections = ();
        $sections{'[Brief]'} = $brief;  # include this section even if blank so we get a title line.
        $sections{'Remarks'} = "$remarks\n" if $remarks ne '';
        $sections{'Syntax'} = $syntax;

        my @params = ();  # have to parse these and build up the wiki tables after, since Markdown needs to know the length of the largest string.  :/

        while (@doxygenlines) {
            my $l = shift @doxygenlines;
            if ($l =~ /\A\\param\s+(.*?)\s+(.*)\Z/) {
                my $arg = $1;
                my $desc = $2;
                while (@doxygenlines) {
                    my $subline = $doxygenlines[0];
                    $subline =~ s/\A\s*//;
                    last if $subline =~ /\A\\/;  # some sort of doxygen command, assume we're past this thing.
                    shift @doxygenlines;  # dump this line from the array; we're using it.
                    if ($subline eq '') {  # empty line, make sure it keeps the newline char.
                        $desc .= "\n";
                    } else {
                        $desc .= " $subline";
                    }
                }

                $desc =~ s/[\s\n]+\Z//ms;

                # We need to know the length of the longest string to make Markdown tables, so we just store these off until everything is parsed.
                push @params, $arg;
                push @params, $desc;
            } elsif ($l =~ /\A\\r(eturns?)\s+(.*)\Z/) {
                my $retstr = "R$1";  # "Return" or "Returns"
                my $desc = $2;
                while (@doxygenlines) {
                    my $subline = $doxygenlines[0];
                    $subline =~ s/\A\s*//;
                    last if $subline =~ /\A\\/;  # some sort of doxygen command, assume we're past this thing.
                    shift @doxygenlines;  # dump this line from the array; we're using it.
                    if ($subline eq '') {  # empty line, make sure it keeps the newline char.
                        $desc .= "\n";
                    } else {
                        $desc .= " $subline";
                    }
                }
                $desc =~ s/[\s\n]+\Z//ms;
                $sections{'Return Value'} = wordwrap("$retstr " . wikify($wikitype, $desc)) . "\n";
            } elsif ($l =~ /\A\\deprecated\s+(.*)\Z/) {
                my $desc = $1;
                while (@doxygenlines) {
                    my $subline = $doxygenlines[0];
                    $subline =~ s/\A\s*//;
                    last if $subline =~ /\A\\/;  # some sort of doxygen command, assume we're past this thing.
                    shift @doxygenlines;  # dump this line from the array; we're using it.
                    if ($subline eq '') {  # empty line, make sure it keeps the newline char.
                        $desc .= "\n";
                    } else {
                        $desc .= " $subline";
                    }
                }
                $desc =~ s/[\s\n]+\Z//ms;
                $sections{'Deprecated'} = wordwrap(wikify($wikitype, $desc)) . "\n";
            } elsif ($l =~ /\A\\since\s+(.*)\Z/) {
                my $desc = $1;
                while (@doxygenlines) {
                    my $subline = $doxygenlines[0];
                    $subline =~ s/\A\s*//;
                    last if $subline =~ /\A\\/;  # some sort of doxygen command, assume we're past this thing.
                    shift @doxygenlines;  # dump this line from the array; we're using it.
                    if ($subline eq '') {  # empty line, make sure it keeps the newline char.
                        $desc .= "\n";
                    } else {
                        $desc .= " $subline";
                    }
                }
                $desc =~ s/[\s\n]+\Z//ms;
                $sections{'Version'} = wordwrap(wikify($wikitype, $desc)) . "\n";
            } elsif ($l =~ /\A\\threadsafety\s+(.*)\Z/) {
                my $desc = $1;
                while (@doxygenlines) {
                    my $subline = $doxygenlines[0];
                    $subline =~ s/\A\s*//;
                    last if $subline =~ /\A\\/;  # some sort of doxygen command, assume we're past this thing.
                    shift @doxygenlines;  # dump this line from the array; we're using it.
                    if ($subline eq '') {  # empty line, make sure it keeps the newline char.
                        $desc .= "\n";
                    } else {
                        $desc .= " $subline";
                    }
                }
                $desc =~ s/[\s\n]+\Z//ms;
                $sections{'Thread Safety'} = wordwrap(wikify($wikitype, $desc)) . "\n";
            } elsif ($l =~ /\A\\sa\s+(.*)\Z/) {
                my $sa = $1;
                $sa =~ s/\(\)\Z//;  # Convert "SDL_Func()" to "SDL_Func"
                $sections{'Related Functions'} = '' if not defined $sections{'Related Functions'};
                if ($wikitype eq 'mediawiki') {
                    $sections{'Related Functions'} .= ":[[$sa]]\n";
                } elsif ($wikitype eq 'md') {
                    $sections{'Related Functions'} .= "* [$sa]($sa)\n";
                } else { die("Expected wikitype '$wikitype'\n"); }
            }
        }

        # Make sure this ends with a double-newline.
        $sections{'Related Functions'} .= "\n" if defined $sections{'Related Functions'};

        # We can build the wiki table now that we have all the data.
        if (scalar(@params) > 0) {
            my $str = '';
            if ($wikitype eq 'mediawiki') {
                while (scalar(@params) > 0) {
                    my $arg = shift @params;
                    my $desc = wikify($wikitype, shift @params);
                    $str .= ($str eq '') ? "{|\n" : "|-\n";
                    $str .= "|'''$arg'''\n";
                    $str .= "|$desc\n";
                }
                $str .= "|}\n";
            } elsif ($wikitype eq 'md') {
                my $longest_arg = 0;
                my $longest_desc = 0;
                my $which = 0;
                foreach (@params) {
                    if ($which == 0) {
                        my $len = length($_) + 4;
                        $longest_arg = $len if ($len > $longest_arg);
                        $which = 1;
                    } else {
                        my $len = length(wikify($wikitype, $_));
                        $longest_desc = $len if ($len > $longest_desc);
                        $which = 0;
                    }
                }

                # Markdown tables are sort of obnoxious.
                $str .= '| ' . (' ' x ($longest_arg+4)) . ' | ' . (' ' x $longest_desc) . " |\n";
                $str .= '| ' . ('-' x ($longest_arg+4)) . ' | ' . ('-' x $longest_desc) . " |\n";

                while (@params) {
                    my $arg = shift @params;
                    my $desc = wikify($wikitype, shift @params);
                    $str .= "| **$arg** " . (' ' x ($longest_arg - length($arg))) . "| $desc" . (' ' x ($longest_desc - length($desc))) . " |\n";
                }
            } else {
                die("Unexpected wikitype!\n");  # should have checked this elsewhere.
            }
            $sections{'Function Parameters'} = $str;
        }

        my $path = "$wikipath/$_.${wikitype}.tmp";
        open(FH, '>', $path) or die("Can't open '$path': $!\n");

        my $sectionsref = $wikifuncs{$fn};

        foreach (@standard_wiki_sections) {
            # drop sections we either replaced or removed from the original wiki's contents.
            if (not defined $only_wiki_sections{$_}) {
                delete($$sectionsref{$_});
            }
        }

        my $wikisectionorderref = $wikisectionorder{$fn};

        # Make sure there's a footer in the wiki that puts this function in CategoryAPI...
        if (not $$sectionsref{'[footer]'}) {
            $$sectionsref{'[footer]'} = '';
            push @$wikisectionorderref, '[footer]';
        }

        # If changing format, convert things that otherwise are passed through unmolested.
        if (defined $changeformat) {
            if (($dewikify_mode eq 'md') and ($origwikitype eq 'mediawiki')) {
                $$sectionsref{'[footer]'} =~ s/\[\[(Category[a-zA-Z0-9_]+)\]\]/[$1]($1)/g;
            } elsif (($dewikify_mode eq 'mediawiki') and ($origwikitype eq 'md')) {
                $$sectionsref{'[footer]'} =~ s/\[(Category[a-zA-Z0-9_]+)\]\(.*?\)/[[$1]]/g;
            }

            foreach (keys %only_wiki_sections) {
                my $sect = $_;
                if (defined $$sectionsref{$sect}) {
                    $$sectionsref{$sect} = wikify($wikitype, dewikify($origwikitype, $$sectionsref{$sect}));
                }
            }
        }

        # !!! FIXME: This won't be CategoryAPI if we eventually handle things other than functions.
        my $footer = $$sectionsref{'[footer]'};

        if ($wikitype eq 'mediawiki') {
            $footer =~ s/\[\[CategoryAPI\]\],?\s*//g;
            $footer = '[[CategoryAPI]]' . (($footer eq '') ? "\n" : ", $footer");
        } elsif ($wikitype eq 'md') {
            $footer =~ s/\[CategoryAPI\]\(CategoryAPI\),?\s*//g;
            $footer = '[CategoryAPI](CategoryAPI)' . (($footer eq '') ? '' : ', ') . $footer;
        } else { die("Unexpected wikitype '$wikitype'\n"); }
        $$sectionsref{'[footer]'} = $footer;

        if (defined $wikipreamble) {
            my $wikified_preamble = wikify($wikitype, $wikipreamble);
            if ($wikitype eq 'mediawiki') {
                print FH "====== $wikified_preamble ======\n";
            } elsif ($wikitype eq 'md') {
                print FH "###### $wikified_preamble\n";
            } else { die("Unexpected wikitype '$wikitype'\n"); }
        }

        my $prevsectstr = '';
        my @ordered_sections = (@standard_wiki_sections, defined $wikisectionorderref ? @$wikisectionorderref : ());  # this copies the arrays into one.
        foreach (@ordered_sections) {
            my $sect = $_;
            next if $sect eq '[start]';
            next if (not defined $sections{$sect} and not defined $$sectionsref{$sect});
            my $section = defined $sections{$sect} ? $sections{$sect} : $$sectionsref{$sect};
            if ($sect eq '[footer]') {
                # Make sure previous section ends with two newlines.
                if (substr($prevsectstr, -1) ne "\n") {
                    print FH "\n\n";
                } elsif (substr($prevsectstr, -2) ne "\n\n") {
                    print FH "\n";
                }
                print FH "----\n";   # It's the same in Markdown and MediaWiki.
            } elsif ($sect eq '[Brief]') {
                if ($wikitype eq 'mediawiki') {
                    print FH  "= $fn =\n\n";
                } elsif ($wikitype eq 'md') {
                    print FH "# $fn\n\n";
                } else { die("Unexpected wikitype '$wikitype'\n"); }
            } else {
                if ($wikitype eq 'mediawiki') {
                    print FH  "\n== $sect ==\n\n";
                } elsif ($wikitype eq 'md') {
                    print FH "\n## $sect\n\n";
                } else { die("Unexpected wikitype '$wikitype'\n"); }
            }

            my $sectstr = defined $sections{$sect} ? $sections{$sect} : $$sectionsref{$sect};
            print FH $sectstr;

            $prevsectstr = $sectstr;

            # make sure these don't show up twice.
            delete($sections{$sect});
            delete($$sectionsref{$sect});
        }

        print FH "\n\n";
        close(FH);

        if (defined $changeformat and ($origwikitype ne $wikitype)) {
            system("cd '$wikipath' ; git mv '$_.${origwikitype}' '$_.${wikitype}'");
            unlink("$wikipath/$_.${origwikitype}");
        }

        rename($path, "$wikipath/$_.${wikitype}") or die("Can't rename '$path' to '$wikipath/$_.${wikitype}': $!\n");
    }

    if (defined $readmepath) {
        if ( -d $readmepath ) {
            mkdir($wikireadmepath);  # just in case
            opendir(DH, $readmepath) or die("Can't opendir '$readmepath': $!\n");
            while (my $d = readdir(DH)) {
                my $dent = $d;
                if ($dent =~ /\AREADME\-(.*?\.md)\Z/) {  # we only bridge Markdown files here.
                    my $wikifname = $1;
                    next if $wikifname eq 'FrontPage.md';
                    filecopy("$readmepath/$dent", "$wikireadmepath/$wikifname", "\n");
                }
            }
            closedir(DH);

            my @pages = ();
            opendir(DH, $wikireadmepath) or die("Can't opendir '$wikireadmepath': $!\n");
            while (my $d = readdir(DH)) {
                my $dent = $d;
                if ($dent =~ /\A(.*?)\.(mediawiki|md)\Z/) {
                    my $wikiname = $1;
                    next if $wikiname eq 'FrontPage';
                    push @pages, $wikiname;
                }
            }
            closedir(DH);

            open(FH, '>', "$wikireadmepath/FrontPage.md") or die("Can't open '$wikireadmepath/FrontPage.md': $!\n");
            print FH "# All READMEs available here\n\n";
            foreach (sort @pages) {
                my $wikiname = $_;
                print FH "- [$wikiname]($wikiname)\n";
            }
            close(FH);
        }
    }

} elsif ($copy_direction == -2) { # --copy-to-manpages
    # This only takes from the wiki data, since it has sections we omit from the headers, like code examples.

    my $manpath = "$srcpath/man";
    mkdir($manpath);
    $manpath .= "/man3";
    mkdir($manpath);

    $dewikify_mode = 'manpage';
    $wordwrap_mode = 'manpage';

    my $introtxt = '';
    if (0) {
    open(FH, '<', "$srcpath/LICENSE.txt") or die("Can't open '$srcpath/LICENSE.txt': $!\n");
    while (<FH>) {
        chomp;
        $introtxt .= ".\\\" $_\n";
    }
    close(FH);
    }

    my $gitrev = `cd "$srcpath" ; git rev-list HEAD~..`;
    chomp($gitrev);

    # !!! FIXME
    open(FH, '<', "$srcpath/$versionfname") or die("Can't open '$srcpath/$versionfname': $!\n");
    my $majorver = 0;
    my $minorver = 0;
    my $patchver = 0;
    while (<FH>) {
        chomp;
        if (/$versionmajorregex/) {
            $majorver = int($1);
        } elsif (/$versionminorregex/) {
            $minorver = int($1);
        } elsif (/$versionpatchregex/) {
            $patchver = int($1);
        }
    }
    close(FH);
    my $fullversion = "$majorver.$minorver.$patchver";

    foreach (keys %headerfuncs) {
        my $fn = $_;
        next if not defined $wikifuncs{$fn};  # don't have a page for that function, skip it.
        my $wikitype = $wikitypes{$fn};
        my $sectionsref = $wikifuncs{$fn};
        my $remarks = $sectionsref->{'Remarks'};
        my $params = $sectionsref->{'Function Parameters'};
        my $returns = $sectionsref->{'Return Value'};
        my $version = $sectionsref->{'Version'};
        my $threadsafety = $sectionsref->{'Thread Safety'};
        my $related = $sectionsref->{'Related Functions'};
        my $examples = $sectionsref->{'Code Examples'};
        my $deprecated = $sectionsref->{'Deprecated'};
        my $brief = $sectionsref->{'[Brief]'};
        my $decl = $headerdecls{$fn};
        my $str = '';

        $brief = "$brief";
        $brief =~ s/\A[\s\n]*\= .*? \=\s*?\n+//ms;
        $brief =~ s/\A[\s\n]*\=\= .*? \=\=\s*?\n+//ms;
        $brief =~ s/\A(.*?\.) /$1\n/;  # \brief should only be one sentence, delimited by a period+space. Split if necessary.
        my @briefsplit = split /\n/, $brief;
        $brief = shift @briefsplit;
        $brief = dewikify($wikitype, $brief);

        if (defined $remarks) {
            $remarks = dewikify($wikitype, join("\n", @briefsplit) . $remarks);
        }

        $str .= $introtxt;

        $str .= ".\\\" This manpage content is licensed under Creative Commons\n";
        $str .= ".\\\"  Attribution 4.0 International (CC BY 4.0)\n";
        $str .= ".\\\"   https://creativecommons.org/licenses/by/4.0/\n";
        $str .= ".\\\" This manpage was generated from ${projectshortname}'s wiki page for $fn:\n";
        $str .= ".\\\"   $wikiurl/$fn\n";
        $str .= ".\\\" Generated with SDL/build-scripts/wikiheaders.pl\n";
        $str .= ".\\\"  revision $gitrev\n" if $gitrev ne '';
        $str .= ".\\\" Please report issues in this manpage's content at:\n";
        $str .= ".\\\"   $bugreporturl\n";
        $str .= ".\\\" Please report issues in the generation of this manpage from the wiki at:\n";
        $str .= ".\\\"   https://github.com/libsdl-org/SDL/issues/new?title=Misgenerated%20manpage%20for%20$fn\n";
        $str .= ".\\\" $projectshortname can be found at $projecturl\n";

        # Define a .URL macro. The "www.tmac" thing decides if we're using GNU roff (which has a .URL macro already), and if so, overrides the macro we just created.
        # This wizadry is from https://web.archive.org/web/20060102165607/http://people.debian.org/~branden/talks/wtfm/wtfm.pdf
        $str .= ".de URL\n";
        $str .= '\\$2 \(laURL: \\$1 \(ra\\$3' . "\n";
        $str .= "..\n";
        $str .= '.if \n[.g] .mso www.tmac' . "\n";

        $str .= ".TH $fn 3 \"$projectshortname $fullversion\" \"$projectfullname\" \"$projectshortname$majorver FUNCTIONS\"\n";
        $str .= ".SH NAME\n";

        $str .= "$fn";
        $str .= " \\- $brief" if (defined $brief);
        $str .= "\n";

        $str .= ".SH SYNOPSIS\n";
        $str .= ".nf\n";
        $str .= ".B #include \\(dq$mainincludefname\\(dq\n";
        $str .= ".PP\n";

        my @decllines = split /\n/, $decl;
        foreach (@decllines) {
            $str .= ".BI \"$_\n";
        }
        $str .= ".fi\n";

        if (defined $remarks) {
            $str .= ".SH DESCRIPTION\n";
            $str .= $remarks . "\n";
        }

        if (defined $deprecated) {
            $str .= ".SH DEPRECATED\n";
            $str .= dewikify($wikitype, $deprecated) . "\n";
        }

        if (defined $params) {
            $str .= ".SH FUNCTION PARAMETERS\n";
            my @lines = split /\n/, $params;
            if ($wikitype eq 'mediawiki') {
                die("Unexpected data parsing MediaWiki table") if (shift @lines ne '{|');  # Dump the '{|' start
                while (scalar(@lines) >= 3) {
                    my $name = shift @lines;
                    my $desc = shift @lines;
                    my $terminator = shift @lines;  # the '|-' or '|}' line.
                    last if ($terminator ne '|-') and ($terminator ne '|}');  # we seem to have run out of table.
                    $name =~ s/\A\|\s*//;
                    $name =~ s/\A\*\*(.*?)\*\*/$1/;
                    $name =~ s/\A\'\'\'(.*?)\'\'\'/$1/;
                    $desc =~ s/\A\|\s*//;
                    $desc = dewikify($wikitype, $desc);
                    #print STDERR "FN: $fn   NAME: $name   DESC: $desc TERM: $terminator\n";

                    $str .= ".TP\n";
                    $str .= ".I $name\n";
                    $str .= "$desc\n";
                }
            } elsif ($wikitype eq 'md') {
                my $l;
                $l = shift @lines;
                die("Unexpected data parsing Markdown table") if (not $l =~ /\A\s*\|\s*\|\s*\|\s*\Z/);
                $l = shift @lines;
                die("Unexpected data parsing Markdown table") if (not $l =~ /\A\s*\|\s*\-*\s*\|\s*\-*\s*\|\s*\Z/);
                while (scalar(@lines) >= 1) {
                    $l = shift @lines;
                    if ($l =~ /\A\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*\Z/) {
                        my $name = $1;
                        my $desc = $2;
                        $name =~ s/\A\*\*(.*?)\*\*/$1/;
                        $name =~ s/\A\'\'\'(.*?)\'\'\'/$1/;
                        $desc = dewikify($wikitype, $desc);

                        $str .= ".TP\n";
                        $str .= ".I $name\n";
                        $str .= "$desc\n";
                    } else {
                        last;  # we seem to have run out of table.
                    }
                }
            } else {
                die("write me");
            }
        }

        if (defined $returns) {
            $str .= ".SH RETURN VALUE\n";
            $str .= dewikify($wikitype, $returns) . "\n";
        }

        if (defined $examples) {
            $str .= ".SH CODE EXAMPLES\n";
            $dewikify_manpage_code_indent = 0;
            $str .= dewikify($wikitype, $examples) . "\n";
            $dewikify_manpage_code_indent = 1;
        }

        if (defined $threadsafety) {
            $str .= ".SH THREAD SAFETY\n";
            $str .= dewikify($wikitype, $threadsafety) . "\n";
        }

        if (defined $version) {
            $str .= ".SH AVAILABILITY\n";
            $str .= dewikify($wikitype, $version) . "\n";
        }

        if (defined $related) {
            $str .= ".SH SEE ALSO\n";
            # !!! FIXME: lots of code duplication in all of these.
            my $v = dewikify($wikitype, $related);
            my @desclines = split /\n/, $v;
            my $nextstr = '';
            foreach (@desclines) {
                s/\A(\:|\* )//;
                s/\(\)\Z//;  # Convert "SDL_Func()" to "SDL_Func"
                s/\[\[(.*?)\]\]/$1/;  # in case some wikilinks remain.
                s/\[(.*?)\]\(.*?\)/$1/;  # in case some wikilinks remain.
                s/\A\*\s*\Z//;
                s/\A\/*//;
                s/\A\.BR\s+//;  # dewikify added this, but we want to handle it.
                s/\A\.I\s+//;  # dewikify added this, but we want to handle it.
                s/\A\s+//;
                s/\s+\Z//;
                next if $_ eq '';
                $str .= "$nextstr.BR $_ (3)";
                $nextstr = ",\n";
            }
            $str .= "\n";
        }

        if (0) {
        $str .= ".SH COPYRIGHT\n";
        $str .= "This manpage is licensed under\n";
        $str .= ".UR https://creativecommons.org/licenses/by/4.0/\n";
        $str .= "Creative Commons Attribution 4.0 International (CC BY 4.0)\n";
        $str .= ".UE\n";
        $str .= ".PP\n";
        $str .= "This manpage was generated from\n";
        $str .= ".UR $wikiurl/$fn\n";
        $str .= "${projectshortname}'s wiki\n";
        $str .= ".UE\n";
        $str .= "using SDL/build-scripts/wikiheaders.pl";
        $str .= " revision $gitrev" if $gitrev ne '';
        $str .= ".\n";
        $str .= "Please report issues in this manpage at\n";
        $str .= ".UR $bugreporturl\n";
        $str .= "our bugtracker!\n";
        $str .= ".UE\n";
        }

        my $path = "$manpath/$_.3.tmp";
        open(FH, '>', $path) or die("Can't open '$path': $!\n");
        print FH $str;
        close(FH);
        rename($path, "$manpath/$_.3") or die("Can't rename '$path' to '$manpath/$_.3': $!\n");
    }
}

# end of wikiheaders.pl ...

