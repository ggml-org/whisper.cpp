#!/usr/bin/perl -w

# To use this script: symlink it to .git/hooks/pre-push, then "git push"
#
# This script is called by "git push" after it has checked the remote status,
# but before anything has been pushed.  If this script exits with a non-zero
# status nothing will be pushed.
#
# This hook is called with the following parameters:
#
# $1 -- Name of the remote to which the push is being done
# $2 -- URL to which the push is being done
#
# If pushing without using a named remote those arguments will be equal.
#
# Information about the commits which are being pushed is supplied as lines to
# the standard input in the form:
#
#   <local ref> <local sha1> <remote ref> <remote sha1>

use warnings;
use strict;

my $remote = $ARGV[0];
my $url = $ARGV[1];

#print("remote: $remote\n");
#print("url: $url\n");

$url =~ s/\.git$//;  # change myorg/myproject.git to myorg/myproject
$url =~ s#^git\@github\.com\:#https://github.com/#i;
my $commiturl = $url =~ /\Ahttps?:\/\/github.com\// ? "$url/commit/" : '';

my $z40 = '0000000000000000000000000000000000000000';
my $reported = 0;

while (<STDIN>) {
    chomp;
    my ($local_ref, $local_sha, $remote_ref, $remote_sha) = split / /;
    #print("local_ref: $local_ref\n");
    #print("local_sha: $local_sha\n");
    #print("remote_ref: $remote_ref\n");
    #print("remote_sha: $remote_sha\n");

    my $range = '';
    if ($remote_sha eq $z40) {  # New branch, examine all commits
        $range = $local_sha;
    } else { # Update to existing branch, examine new commits
        $range = "$remote_sha..$local_sha";
    }

    my $gitcmd = "git log --reverse --oneline --no-abbrev-commit '$range'";
    open(GITPIPE, '-|', $gitcmd) or die("\n\n$0: Failed to run '$gitcmd': $!\n\nAbort push!\n\n");
    while (<GITPIPE>) {
        chomp;
        if (/\A([a-fA-F0-9]+)\s+(.*?)\Z/) {
            my $hash = $1;
            my $msg = $2;

            if (!$reported) {
                print("\nCommits expected to be pushed:\n");
                $reported = 1;
            }

            #print("hash: $hash\n");
            #print("msg: $msg\n");

            print("$commiturl$hash -- $msg\n");
        } else {
            die("$0: Unexpected output from '$gitcmd'!\n\nAbort push!\n\n");
        }
    }
    die("\n\n$0: Failing exit code from running '$gitcmd'!\n\nAbort push!\n\n") if !close(GITPIPE);
}

print("\n") if $reported;

exit(0);  # Let the push go forward.

# vi: set ts=4 sw=4 expandtab:
