/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/*

 Used by the test framework and test cases.

*/

#include "SDL_config.h"

#include "SDL_test.h"

/* Assert check message format */
#define SDLTEST_ASSERT_CHECK_FORMAT "Assert '%s': %s"

/* Assert summary message format */
#define SDLTEST_ASSERT_SUMMARY_FORMAT "Assert Summary: Total=%d Passed=%d Failed=%d"

/* ! \brief counts the failed asserts */
static int SDLTest_AssertsFailed = 0;

/* ! \brief counts the passed asserts */
static int SDLTest_AssertsPassed = 0;

/*
 *  Assert that logs and break execution flow on failures (i.e. for harness errors).
 */
void SDLTest_Assert(int assertCondition, SDL_PRINTF_FORMAT_STRING const char *assertDescription, ...)
{
    va_list list;
    char logMessage[SDLTEST_MAX_LOGMESSAGE_LENGTH];

    /* Print assert description into a buffer */
    SDL_memset(logMessage, 0, SDLTEST_MAX_LOGMESSAGE_LENGTH);
    va_start(list, assertDescription);
    (void)SDL_vsnprintf(logMessage, SDLTEST_MAX_LOGMESSAGE_LENGTH - 1, assertDescription, list);
    va_end(list);

    /* Log, then assert and break on failure */
    SDL_assert((SDLTest_AssertCheck(assertCondition, "%s", logMessage)));
}

/*
 * Assert that logs but does not break execution flow on failures (i.e. for test cases).
 */
int SDLTest_AssertCheck(int assertCondition, SDL_PRINTF_FORMAT_STRING const char *assertDescription, ...)
{
    va_list list;
    char logMessage[SDLTEST_MAX_LOGMESSAGE_LENGTH];

    /* Print assert description into a buffer */
    SDL_memset(logMessage, 0, SDLTEST_MAX_LOGMESSAGE_LENGTH);
    va_start(list, assertDescription);
    (void)SDL_vsnprintf(logMessage, SDLTEST_MAX_LOGMESSAGE_LENGTH - 1, assertDescription, list);
    va_end(list);

    /* Log pass or fail message */
    if (assertCondition == ASSERT_FAIL) {
        SDLTest_AssertsFailed++;
        SDLTest_LogError(SDLTEST_ASSERT_CHECK_FORMAT, logMessage, "Failed");
    } else {
        SDLTest_AssertsPassed++;
        SDLTest_Log(SDLTEST_ASSERT_CHECK_FORMAT, logMessage, "Passed");
    }

    return assertCondition;
}

/*
 * Explicitly passing Assert that logs (i.e. for test cases).
 */
void SDLTest_AssertPass(SDL_PRINTF_FORMAT_STRING const char *assertDescription, ...)
{
    va_list list;
    char logMessage[SDLTEST_MAX_LOGMESSAGE_LENGTH];

    /* Print assert description into a buffer */
    SDL_memset(logMessage, 0, SDLTEST_MAX_LOGMESSAGE_LENGTH);
    va_start(list, assertDescription);
    (void)SDL_vsnprintf(logMessage, SDLTEST_MAX_LOGMESSAGE_LENGTH - 1, assertDescription, list);
    va_end(list);

    /* Log pass message */
    SDLTest_AssertsPassed++;
    SDLTest_Log(SDLTEST_ASSERT_CHECK_FORMAT, logMessage, "Passed");
}

/*
 * Resets the assert summary counters to zero.
 */
void SDLTest_ResetAssertSummary(void)
{
    SDLTest_AssertsPassed = 0;
    SDLTest_AssertsFailed = 0;
}

/*
 * Logs summary of all assertions (total, pass, fail) since last reset
 * as INFO (failed==0) or ERROR (failed > 0).
 */
void SDLTest_LogAssertSummary(void)
{
    int totalAsserts = SDLTest_AssertsPassed + SDLTest_AssertsFailed;
    if (SDLTest_AssertsFailed == 0) {
        SDLTest_Log(SDLTEST_ASSERT_SUMMARY_FORMAT, totalAsserts, SDLTest_AssertsPassed, SDLTest_AssertsFailed);
    } else {
        SDLTest_LogError(SDLTEST_ASSERT_SUMMARY_FORMAT, totalAsserts, SDLTest_AssertsPassed, SDLTest_AssertsFailed);
    }
}

/*
 * Converts the current assert state into a test result
 */
int SDLTest_AssertSummaryToTestResult(void)
{
    if (SDLTest_AssertsFailed > 0) {
        return TEST_RESULT_FAILED;
    } else {
        if (SDLTest_AssertsPassed > 0) {
            return TEST_RESULT_PASSED;
        } else {
            return TEST_RESULT_NO_ASSERT;
        }
    }
}

/* vi: set ts=4 sw=4 expandtab: */
