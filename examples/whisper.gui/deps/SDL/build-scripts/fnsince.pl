#!/usr/bin/perl -w

use warnings;
use strict;
use File::Basename;
use Cwd qw(abs_path);

my $wikipath = undef;
foreach (@ARGV) {
    $wikipath = abs_path($_), next if not defined $wikipath;
}

chdir(dirname(__FILE__));
chdir('..');

my @unsorted_releases = ();
open(PIPEFH, '-|', 'git tag -l') or die "Failed to read git release tags: $!\n";

while (<PIPEFH>) {
    chomp;
    if (/\Arelease\-(.*?)\Z/) {
        # After 2.24.x, ignore anything that isn't a x.y.0 release.
        # We moved to bugfix-only point releases there, so make sure new APIs
        #  are assigned to the next minor version and ignore the patch versions.
        my $ver = $1;
        my @versplit = split /\./, $ver;
        next if (scalar(@versplit) > 2) && (($versplit[0] > 2) || (($versplit[0] == 2) && ($versplit[1] >= 24))) && ($versplit[2] != 0);

        # Consider this release version.
        push @unsorted_releases, $ver;
    }

}
close(PIPEFH);

#print("\n\nUNSORTED\n");
#foreach (@unsorted_releases) {
#    print "$_\n";
#}

my @releases = sort {
    my @asplit = split /\./, $a;
    my @bsplit = split /\./, $b;
    my $rc;
    for (my $i = 0; $i < scalar(@asplit); $i++) {
        return 1 if (scalar(@bsplit) <= $i);  # a is "2.0.1" and b is "2.0", or whatever.
        my $aseg = $asplit[$i];
        my $bseg = $bsplit[$i];
        $rc = int($aseg) <=> int($bseg);
        return $rc if ($rc != 0);  # found the difference.
    }
    return 0;  # still here? They matched completely?!
} @unsorted_releases;

# this happens to work for how SDL versions things at the moment.
my $current_release = $releases[-1];
my $next_release;

if ($current_release eq '2.0.22') {  # Hack for our jump from 2.0.22 to 2.24.0...
    $next_release = '2.24.0';
} else {
    my @current_release_segments = split /\./, $current_release;
    @current_release_segments[1] = '' . ($current_release_segments[1] + 2);
    $next_release = join('.', @current_release_segments);
}

#print("\n\nSORTED\n");
#foreach (@releases) {
#    print "$_\n";
#}
#print("\nCURRENT RELEASE: $current_release\n");
#print("NEXT RELEASE: $next_release\n\n");

push @releases, 'HEAD';

my %funcs = ();
foreach my $release (@releases) {
    #print("Checking $release...\n");
    next if ($release eq '2.0.0') || ($release eq '2.0.1');  # no dynapi before 2.0.2
    my $assigned_release = ($release eq '2.0.2') ? '2.0.0' : $release;  # assume everything in 2.0.2--first with dynapi--was there since 2.0.0. We'll fix it up later.
    my $tag = ($release eq 'HEAD') ? $release : "release-$release";
    my $blobname = "$tag:src/dynapi/SDL_dynapi_overrides.h";
    open(PIPEFH, '-|', "git show '$blobname'") or die "Failed to read git blob '$blobname': $!\n";
    while (<PIPEFH>) {
        chomp;
        if (/\A\#define\s+(SDL_.*?)\s+SDL_.*?_REAL\Z/) {
            my $fn = $1;
            $funcs{$fn} = $assigned_release if not defined $funcs{$fn};
        }
    }
    close(PIPEFH);
}

# Fixup the handful of functions that were added in 2.0.1 and 2.0.2 that we
#  didn't have dynapi revision data about...
$funcs{'SDL_GetSystemRAM'} = '2.0.1';
$funcs{'SDL_GetBasePath'} = '2.0.1';
$funcs{'SDL_GetPrefPath'} = '2.0.1';
$funcs{'SDL_UpdateYUVTexture'} = '2.0.1';
$funcs{'SDL_GL_GetDrawableSize'} = '2.0.1';
$funcs{'SDL_Direct3D9GetAdapterIndex'} = '2.0.1';
$funcs{'SDL_RenderGetD3D9Device'} = '2.0.1';

$funcs{'SDL_RegisterApp'} = '2.0.2';
$funcs{'SDL_UnregisterApp'} = '2.0.2';
$funcs{'SDL_GetAssertionHandler'} = '2.0.2';
$funcs{'SDL_GetDefaultAssertionHandler'} = '2.0.2';
$funcs{'SDL_AtomicAdd'} = '2.0.2';
$funcs{'SDL_AtomicGet'} = '2.0.2';
$funcs{'SDL_AtomicGetPtr'} = '2.0.2';
$funcs{'SDL_AtomicSet'} = '2.0.2';
$funcs{'SDL_AtomicSetPtr'} = '2.0.2';
$funcs{'SDL_HasAVX'} = '2.0.2';
$funcs{'SDL_GameControllerAddMappingsFromRW'} = '2.0.2';
$funcs{'SDL_acos'} = '2.0.2';
$funcs{'SDL_asin'} = '2.0.2';
$funcs{'SDL_vsscanf'} = '2.0.2';
$funcs{'SDL_DetachThread'} = '2.0.2';
$funcs{'SDL_GL_ResetAttributes'} = '2.0.2';
$funcs{'SDL_DXGIGetOutputInfo'} = '2.0.2';

# these are incorrect in the dynapi header, because we forgot to add them
#  until a later release, but are available in the older release.
$funcs{'SDL_WinRTGetFSPathUNICODE'} = '2.0.3';
$funcs{'SDL_WinRTGetFSPathUTF8'} = '2.0.3';
$funcs{'SDL_WinRTRunApp'} = '2.0.3';

if (not defined $wikipath) {
    foreach my $release (@releases) {
        foreach my $fn (sort keys %funcs) {
            print("$fn: $funcs{$fn}\n") if $funcs{$fn} eq $release;
        }
    }
} else {
    if (defined $wikipath) {
        chdir($wikipath);
        foreach my $fn (keys %funcs) {
            my $revision = $funcs{$fn};
            $revision = $next_release if $revision eq 'HEAD';
            my $fname = "$fn.mediawiki";
            if ( ! -f $fname ) {
                #print STDERR "No such file: $fname\n";
                next;
            }

            my @lines = ();
            open(FH, '<', $fname) or die("Can't open $fname for read: $!\n");
            my $added = 0;
            while (<FH>) {
                chomp;
                if ((/\A\-\-\-\-/) && (!$added)) {
                    push @lines, "== Version ==";
                    push @lines, "";
                    push @lines, "This function is available since SDL $revision.";
                    push @lines, "";
                    $added = 1;
                }
                push @lines, $_;
                next if not /\A\=\=\s+Version\s+\=\=/;
                $added = 1;
                push @lines, "";
                push @lines, "This function is available since SDL $revision.";
                push @lines, "";
                while (<FH>) {
                    chomp;
                    next if not (/\A\=\=\s+/ || /\A\-\-\-\-/);
                    push @lines, $_;
                    last;
                }
            }
            close(FH);

            if (!$added) {
                push @lines, "== Version ==";
                push @lines, "";
                push @lines, "This function is available since SDL $revision.";
                push @lines, "";
            }

            open(FH, '>', $fname) or die("Can't open $fname for write: $!\n");
            foreach (@lines) {
                print FH "$_\n";
            }
            close(FH);
        }
    }
}

