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

#include "SDL_config.h"

#include "SDL_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Invalid test name/description message format */
#define SDLTEST_INVALID_NAME_FORMAT "(Invalid)"

/* Log summary message format */
#define SDLTEST_LOG_SUMMARY_FORMAT "%s Summary: Total=%d Passed=%d Failed=%d Skipped=%d"

/* Final result message format */
#define SDLTEST_FINAL_RESULT_FORMAT ">>> %s '%s': %s\n"

/* ! \brief Timeout for single test case execution */
static Uint32 SDLTest_TestCaseTimeout = 3600;

/**
 * Generates a random run seed string for the harness. The generated seed
 * will contain alphanumeric characters (0-9A-Z).
 *
 * Note: The returned string needs to be deallocated by the caller.
 *
 * \param length The length of the seed string to generate
 *
 * \returns The generated seed string
 */
char *SDLTest_GenerateRunSeed(const int length)
{
    char *seed = NULL;
    SDLTest_RandomContext randomContext;
    int counter;

    /* Sanity check input */
    if (length <= 0) {
        SDLTest_LogError("The length of the harness seed must be >0.");
        return NULL;
    }

    /* Allocate output buffer */
    seed = (char *)SDL_malloc((length + 1) * sizeof(char));
    if (!seed) {
        SDLTest_LogError("SDL_malloc for run seed output buffer failed.");
        SDL_Error(SDL_ENOMEM);
        return NULL;
    }

    /* Generate a random string of alphanumeric characters */
    SDLTest_RandomInitTime(&randomContext);
    for (counter = 0; counter < length; counter++) {
        unsigned int number = SDLTest_Random(&randomContext);
        char ch = (char)(number % (91 - 48)) + 48;
        if (ch >= 58 && ch <= 64) {
            ch = 65;
        }
        seed[counter] = ch;
    }
    seed[length] = '\0';

    return seed;
}

/**
 * Generates an execution key for the fuzzer.
 *
 * \param runSeed        The run seed to use
 * \param suiteName      The name of the test suite
 * \param testName       The name of the test
 * \param iteration      The iteration count
 *
 * \returns The generated execution key to initialize the fuzzer with.
 *
 */
static Uint64 SDLTest_GenerateExecKey(const char *runSeed, const char *suiteName, const char *testName, int iteration)
{
    SDLTest_Md5Context md5Context;
    Uint64 *keys;
    char iterationString[16];
    size_t runSeedLength;
    size_t suiteNameLength;
    size_t testNameLength;
    size_t iterationStringLength;
    size_t entireStringLength;
    char *buffer;

    if (!runSeed || runSeed[0] == '\0') {
        SDLTest_LogError("Invalid runSeed string.");
        return -1;
    }

    if (!suiteName || suiteName[0] == '\0') {
        SDLTest_LogError("Invalid suiteName string.");
        return -1;
    }

    if (!testName || testName[0] == '\0') {
        SDLTest_LogError("Invalid testName string.");
        return -1;
    }

    if (iteration <= 0) {
        SDLTest_LogError("Invalid iteration count.");
        return -1;
    }

    /* Convert iteration number into a string */
    SDL_memset(iterationString, 0, sizeof(iterationString));
    (void)SDL_snprintf(iterationString, sizeof(iterationString) - 1, "%d", iteration);

    /* Combine the parameters into single string */
    runSeedLength = SDL_strlen(runSeed);
    suiteNameLength = SDL_strlen(suiteName);
    testNameLength = SDL_strlen(testName);
    iterationStringLength = SDL_strlen(iterationString);
    entireStringLength = runSeedLength + suiteNameLength + testNameLength + iterationStringLength + 1;
    buffer = (char *)SDL_malloc(entireStringLength);
    if (!buffer) {
        SDLTest_LogError("Failed to allocate buffer for execKey generation.");
        SDL_Error(SDL_ENOMEM);
        return 0;
    }
    (void)SDL_snprintf(buffer, entireStringLength, "%s%s%s%d", runSeed, suiteName, testName, iteration);

    /* Hash string and use half of the digest as 64bit exec key */
    SDLTest_Md5Init(&md5Context);
    SDLTest_Md5Update(&md5Context, (unsigned char *)buffer, (unsigned int)entireStringLength);
    SDLTest_Md5Final(&md5Context);
    SDL_free(buffer);
    keys = (Uint64 *)md5Context.digest;

    return keys[0];
}

/**
 * \brief Set timeout handler for test.
 *
 * Note: SDL_Init(SDL_INIT_TIMER) will be called if it wasn't done so before.
 *
 * \param timeout Timeout interval in seconds.
 * \param callback Function that will be called after timeout has elapsed.
 *
 * \return Timer id or -1 on failure.
 */
static SDL_TimerID SDLTest_SetTestTimeout(int timeout, void(SDLCALL *callback)(void))
{
    Uint32 timeoutInMilliseconds;
    SDL_TimerID timerID;

    if (!callback) {
        SDLTest_LogError("Timeout callback can't be NULL");
        return -1;
    }

    if (timeout < 0) {
        SDLTest_LogError("Timeout value must be bigger than zero.");
        return -1;
    }

    /* Init SDL timer if not initialized before */
    if (SDL_WasInit(SDL_INIT_TIMER) == 0) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            SDLTest_LogError("Failed to init timer subsystem: %s", SDL_GetError());
            return -1;
        }
    }

    /* Set timer */
    timeoutInMilliseconds = timeout * 1000;
    timerID = SDL_AddTimer(timeoutInMilliseconds, (SDL_TimerCallback)callback, 0x0);
    if (timerID == 0) {
        SDLTest_LogError("Creation of SDL timer failed: %s", SDL_GetError());
        return -1;
    }

    return timerID;
}

/**
 * \brief Timeout handler. Aborts test run and exits harness process.
 */
#if defined(__WATCOMC__)
#pragma aux SDLTest_BailOut aborts;
#endif
static SDL_NORETURN void SDLCALL SDLTest_BailOut(void)
{
    SDLTest_LogError("TestCaseTimeout timer expired. Aborting test run.");
    exit(TEST_ABORTED); /* bail out from the test */
}

/**
 * \brief Execute a test using the given execution key.
 *
 * \param testSuite Suite containing the test case.
 * \param testCase Case to execute.
 * \param execKey Execution key for the fuzzer.
 * \param forceTestRun Force test to run even if test was disabled in suite.
 *
 * \returns Test case result.
 */
static int SDLTest_RunTest(SDLTest_TestSuiteReference *testSuite, const SDLTest_TestCaseReference *testCase, Uint64 execKey, SDL_bool forceTestRun)
{
    SDL_TimerID timer = 0;
    int testCaseResult = 0;
    int testResult = 0;
    int fuzzerCount;

    if (!testSuite || !testCase || !testSuite->name || !testCase->name) {
        SDLTest_LogError("Setup failure: testSuite or testCase references NULL");
        return TEST_RESULT_SETUP_FAILURE;
    }

    if (!testCase->enabled && forceTestRun == SDL_FALSE) {
        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, "Skipped (Disabled)");
        return TEST_RESULT_SKIPPED;
    }

    /* Initialize fuzzer */
    SDLTest_FuzzerInit(execKey);

    /* Reset assert tracker */
    SDLTest_ResetAssertSummary();

    /* Set timeout timer */
    timer = SDLTest_SetTestTimeout(SDLTest_TestCaseTimeout, SDLTest_BailOut);

    /* Maybe run suite initalizer function */
    if (testSuite->testSetUp) {
        testSuite->testSetUp(0x0);
        if (SDLTest_AssertSummaryToTestResult() == TEST_RESULT_FAILED) {
            SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Suite Setup", testSuite->name, "Failed");
            return TEST_RESULT_SETUP_FAILURE;
        }
    }

    /* Run test case function */
    testCaseResult = testCase->testCase(0x0);

    /* Convert test execution result into harness result */
    if (testCaseResult == TEST_SKIPPED) {
        /* Test was programatically skipped */
        testResult = TEST_RESULT_SKIPPED;
    } else if (testCaseResult == TEST_STARTED) {
        /* Test did not return a TEST_COMPLETED value; assume it failed */
        testResult = TEST_RESULT_FAILED;
    } else if (testCaseResult == TEST_ABORTED) {
        /* Test was aborted early; assume it failed */
        testResult = TEST_RESULT_FAILED;
    } else {
        /* Perform failure analysis based on asserts */
        testResult = SDLTest_AssertSummaryToTestResult();
    }

    /* Maybe run suite cleanup function (ignore failed asserts) */
    if (testSuite->testTearDown) {
        testSuite->testTearDown(0x0);
    }

    /* Cancel timeout timer */
    if (timer) {
        SDL_RemoveTimer(timer);
    }

    /* Report on asserts and fuzzer usage */
    fuzzerCount = SDLTest_GetFuzzerInvocationCount();
    if (fuzzerCount > 0) {
        SDLTest_Log("Fuzzer invocations: %d", fuzzerCount);
    }

    /* Final log based on test execution result */
    if (testCaseResult == TEST_SKIPPED) {
        /* Test was programatically skipped */
        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, "Skipped (Programmatically)");
    } else if (testCaseResult == TEST_STARTED) {
        /* Test did not return a TEST_COMPLETED value; assume it failed */
        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, "Failed (test started, but did not return TEST_COMPLETED)");
    } else if (testCaseResult == TEST_ABORTED) {
        /* Test was aborted early; assume it failed */
        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, "Failed (Aborted)");
    } else {
        SDLTest_LogAssertSummary();
    }

    return testResult;
}

/* Prints summary of all suites/tests contained in the given reference */
#if 0
static void SDLTest_LogTestSuiteSummary(SDLTest_TestSuiteReference *testSuites)
{
    int suiteCounter;
    int testCounter;
    SDLTest_TestSuiteReference *testSuite;
    SDLTest_TestCaseReference *testCase;

    /* Loop over all suites */
    suiteCounter = 0;
    while (&testSuites[suiteCounter]) {
        testSuite=&testSuites[suiteCounter];
        suiteCounter++;
        SDLTest_Log("Test Suite %i - %s\n", suiteCounter,
            (testSuite->name) ? testSuite->name : SDLTEST_INVALID_NAME_FORMAT);

        /* Loop over all test cases */
        testCounter = 0;
        while (testSuite->testCases[testCounter]) {
            testCase=(SDLTest_TestCaseReference *)testSuite->testCases[testCounter];
            testCounter++;
            SDLTest_Log("  Test Case %i - %s: %s", testCounter,
                (testCase->name) ? testCase->name : SDLTEST_INVALID_NAME_FORMAT,
                (testCase->description) ? testCase->description : SDLTEST_INVALID_NAME_FORMAT);
        }
    }
}
#endif

/* Gets a timer value in seconds */
static float GetClock(void)
{
    float currentClock = SDL_GetPerformanceCounter() / (float)SDL_GetPerformanceFrequency();
    return currentClock;
}

/**
 * \brief Execute a test suite using the given run seed and execution key.
 *
 * The filter string is matched to the suite name (full comparison) to select a single suite,
 * or if no suite matches, it is matched to the test names (full comparison) to select a single test.
 *
 * \param testSuites Suites containing the test case.
 * \param userRunSeed Custom run seed provided by user, or NULL to autogenerate one.
 * \param userExecKey Custom execution key provided by user, or 0 to autogenerate one.
 * \param filter Filter specification. NULL disables. Case sensitive.
 * \param testIterations Number of iterations to run each test case.
 *
 * \returns Test run result; 0 when all tests passed, 1 if any tests failed.
 */
int SDLTest_RunSuites(SDLTest_TestSuiteReference *testSuites[], const char *userRunSeed, Uint64 userExecKey, const char *filter, int testIterations)
{
    int totalNumberOfTests = 0;
    int failedNumberOfTests = 0;
    int suiteCounter;
    int testCounter;
    int iterationCounter;
    SDLTest_TestSuiteReference *testSuite;
    const SDLTest_TestCaseReference *testCase;
    const char *runSeed = NULL;
    const char *currentSuiteName;
    const char *currentTestName;
    Uint64 execKey;
    float runStartSeconds;
    float suiteStartSeconds;
    float testStartSeconds;
    float runEndSeconds;
    float suiteEndSeconds;
    float testEndSeconds;
    float runtime;
    int suiteFilter = 0;
    const char *suiteFilterName = NULL;
    int testFilter = 0;
    const char *testFilterName = NULL;
    SDL_bool forceTestRun = SDL_FALSE;
    int testResult = 0;
    int runResult = 0;
    int totalTestFailedCount = 0;
    int totalTestPassedCount = 0;
    int totalTestSkippedCount = 0;
    int testFailedCount = 0;
    int testPassedCount = 0;
    int testSkippedCount = 0;
    int countSum = 0;
    const SDLTest_TestCaseReference **failedTests;
    char generatedSeed[16 + 1];

    /* Sanitize test iterations */
    if (testIterations < 1) {
        testIterations = 1;
    }

    /* Generate run see if we don't have one already */
    if (!userRunSeed || userRunSeed[0] == '\0') {
        char *tmp = SDLTest_GenerateRunSeed(16);
        if (!tmp) {
            SDLTest_LogError("Generating a random seed failed");
            return 2;
        }
        SDL_memcpy(generatedSeed, tmp, 16 + 1);
        SDL_free(tmp);
        runSeed = generatedSeed;
    } else {
        runSeed = userRunSeed;
    }

    /* Reset per-run counters */
    totalTestFailedCount = 0;
    totalTestPassedCount = 0;
    totalTestSkippedCount = 0;

    /* Take time - run start */
    runStartSeconds = GetClock();

    /* Log run with fuzzer parameters */
    SDLTest_Log("::::: Test Run /w seed '%s' started\n", runSeed);

    /* Count the total number of tests */
    suiteCounter = 0;
    while (testSuites[suiteCounter]) {
        testSuite = testSuites[suiteCounter];
        suiteCounter++;
        testCounter = 0;
        while (testSuite->testCases[testCounter]) {
            testCounter++;
            totalNumberOfTests++;
        }
    }

    if (totalNumberOfTests == 0) {
        SDLTest_LogError("No tests to run?");
        return -1;
    }

    /* Pre-allocate an array for tracking failed tests (potentially all test cases) */
    failedTests = (const SDLTest_TestCaseReference **)SDL_malloc(totalNumberOfTests * sizeof(SDLTest_TestCaseReference *));
    if (!failedTests) {
        SDLTest_LogError("Unable to allocate cache for failed tests");
        SDL_Error(SDL_ENOMEM);
        return -1;
    }

    /* Initialize filtering */
    if (filter && filter[0] != '\0') {
        /* Loop over all suites to check if we have a filter match */
        suiteCounter = 0;
        while (testSuites[suiteCounter] && suiteFilter == 0) {
            testSuite = testSuites[suiteCounter];
            suiteCounter++;
            if (testSuite->name && SDL_strcasecmp(filter, testSuite->name) == 0) {
                /* Matched a suite name */
                suiteFilter = 1;
                suiteFilterName = testSuite->name;
                SDLTest_Log("Filtering: running only suite '%s'", suiteFilterName);
                break;
            }

            /* Within each suite, loop over all test cases to check if we have a filter match */
            testCounter = 0;
            while (testSuite->testCases[testCounter] && testFilter == 0) {
                testCase = testSuite->testCases[testCounter];
                testCounter++;
                if (testCase->name && SDL_strcasecmp(filter, testCase->name) == 0) {
                    /* Matched a test name */
                    suiteFilter = 1;
                    suiteFilterName = testSuite->name;
                    testFilter = 1;
                    testFilterName = testCase->name;
                    SDLTest_Log("Filtering: running only test '%s' in suite '%s'", testFilterName, suiteFilterName);
                    break;
                }
            }
        }

        if (suiteFilter == 0 && testFilter == 0) {
            SDLTest_LogError("Filter '%s' did not match any test suite/case.", filter);
            for (suiteCounter = 0; testSuites[suiteCounter]; ++suiteCounter) {
                testSuite = testSuites[suiteCounter];
                if (testSuite->name) {
                    SDLTest_Log("Test suite: %s", testSuite->name);
                }

                /* Within each suite, loop over all test cases to check if we have a filter match */
                for (testCounter = 0; testSuite->testCases[testCounter]; ++testCounter) {
                    testCase = testSuite->testCases[testCounter];
                    SDLTest_Log("      test: %s%s", testCase->name, testCase->enabled ? "" : " (disabled)");
                }
            }
            SDLTest_Log("Exit code: 2");
            SDL_free((void *)failedTests);
            return 2;
        }
    }

    /* Loop over all suites */
    suiteCounter = 0;
    while (testSuites[suiteCounter]) {
        testSuite = testSuites[suiteCounter];
        currentSuiteName = (testSuite->name ? testSuite->name : SDLTEST_INVALID_NAME_FORMAT);
        suiteCounter++;

        /* Filter suite if flag set and we have a name */
        if (suiteFilter == 1 && suiteFilterName && testSuite->name &&
            SDL_strcasecmp(suiteFilterName, testSuite->name) != 0) {
            /* Skip suite */
            SDLTest_Log("===== Test Suite %i: '%s' skipped\n",
                        suiteCounter,
                        currentSuiteName);
        } else {

            /* Reset per-suite counters */
            testFailedCount = 0;
            testPassedCount = 0;
            testSkippedCount = 0;

            /* Take time - suite start */
            suiteStartSeconds = GetClock();

            /* Log suite started */
            SDLTest_Log("===== Test Suite %i: '%s' started\n",
                        suiteCounter,
                        currentSuiteName);

            /* Loop over all test cases */
            testCounter = 0;
            while (testSuite->testCases[testCounter]) {
                testCase = testSuite->testCases[testCounter];
                currentTestName = (testCase->name ? testCase->name : SDLTEST_INVALID_NAME_FORMAT);
                testCounter++;

                /* Filter tests if flag set and we have a name */
                if (testFilter == 1 && testFilterName && testCase->name &&
                    SDL_strcasecmp(testFilterName, testCase->name) != 0) {
                    /* Skip test */
                    SDLTest_Log("===== Test Case %i.%i: '%s' skipped\n",
                                suiteCounter,
                                testCounter,
                                currentTestName);
                } else {
                    /* Override 'disabled' flag if we specified a test filter (i.e. force run for debugging) */
                    if (testFilter == 1 && !testCase->enabled) {
                        SDLTest_Log("Force run of disabled test since test filter was set");
                        forceTestRun = SDL_TRUE;
                    }

                    /* Take time - test start */
                    testStartSeconds = GetClock();

                    /* Log test started */
                    SDLTest_Log("----- Test Case %i.%i: '%s' started",
                                suiteCounter,
                                testCounter,
                                currentTestName);
                    if (testCase->description && testCase->description[0] != '\0') {
                        SDLTest_Log("Test Description: '%s'",
                                    (testCase->description) ? testCase->description : SDLTEST_INVALID_NAME_FORMAT);
                    }

                    /* Loop over all iterations */
                    iterationCounter = 0;
                    while (iterationCounter < testIterations) {
                        iterationCounter++;

                        if (userExecKey != 0) {
                            execKey = userExecKey;
                        } else {
                            execKey = SDLTest_GenerateExecKey(runSeed, testSuite->name, testCase->name, iterationCounter);
                        }

                        SDLTest_Log("Test Iteration %i: execKey %" SDL_PRIu64, iterationCounter, execKey);
                        testResult = SDLTest_RunTest(testSuite, testCase, execKey, forceTestRun);

                        if (testResult == TEST_RESULT_PASSED) {
                            testPassedCount++;
                            totalTestPassedCount++;
                        } else if (testResult == TEST_RESULT_SKIPPED) {
                            testSkippedCount++;
                            totalTestSkippedCount++;
                        } else {
                            testFailedCount++;
                            totalTestFailedCount++;
                        }
                    }

                    /* Take time - test end */
                    testEndSeconds = GetClock();
                    runtime = testEndSeconds - testStartSeconds;
                    if (runtime < 0.0f) {
                        runtime = 0.0f;
                    }

                    if (testIterations > 1) {
                        /* Log test runtime */
                        SDLTest_Log("Runtime of %i iterations: %.1f sec", testIterations, runtime);
                        SDLTest_Log("Average Test runtime: %.5f sec", runtime / (float)testIterations);
                    } else {
                        /* Log test runtime */
                        SDLTest_Log("Total Test runtime: %.1f sec", runtime);
                    }

                    /* Log final test result */
                    switch (testResult) {
                    case TEST_RESULT_PASSED:
                        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Test", currentTestName, "Passed");
                        break;
                    case TEST_RESULT_FAILED:
                        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", currentTestName, "Failed");
                        break;
                    case TEST_RESULT_NO_ASSERT:
                        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", currentTestName, "No Asserts");
                        break;
                    }

                    /* Collect failed test case references for repro-step display */
                    if (testResult == TEST_RESULT_FAILED) {
                        failedTests[failedNumberOfTests] = testCase;
                        failedNumberOfTests++;
                    }
                }
            }

            /* Take time - suite end */
            suiteEndSeconds = GetClock();
            runtime = suiteEndSeconds - suiteStartSeconds;
            if (runtime < 0.0f) {
                runtime = 0.0f;
            }

            /* Log suite runtime */
            SDLTest_Log("Total Suite runtime: %.1f sec", runtime);

            /* Log summary and final Suite result */
            countSum = testPassedCount + testFailedCount + testSkippedCount;
            if (testFailedCount == 0) {
                SDLTest_Log(SDLTEST_LOG_SUMMARY_FORMAT, "Suite", countSum, testPassedCount, testFailedCount, testSkippedCount);
                SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Suite", currentSuiteName, "Passed");
            } else {
                SDLTest_LogError(SDLTEST_LOG_SUMMARY_FORMAT, "Suite", countSum, testPassedCount, testFailedCount, testSkippedCount);
                SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Suite", currentSuiteName, "Failed");
            }
        }
    }

    /* Take time - run end */
    runEndSeconds = GetClock();
    runtime = runEndSeconds - runStartSeconds;
    if (runtime < 0.0f) {
        runtime = 0.0f;
    }

    /* Log total runtime */
    SDLTest_Log("Total Run runtime: %.1f sec", runtime);

    /* Log summary and final run result */
    countSum = totalTestPassedCount + totalTestFailedCount + totalTestSkippedCount;
    if (totalTestFailedCount == 0) {
        runResult = 0;
        SDLTest_Log(SDLTEST_LOG_SUMMARY_FORMAT, "Run", countSum, totalTestPassedCount, totalTestFailedCount, totalTestSkippedCount);
        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Run /w seed", runSeed, "Passed");
    } else {
        runResult = 1;
        SDLTest_LogError(SDLTEST_LOG_SUMMARY_FORMAT, "Run", countSum, totalTestPassedCount, totalTestFailedCount, totalTestSkippedCount);
        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Run /w seed", runSeed, "Failed");
    }

    /* Print repro steps for failed tests */
    if (failedNumberOfTests > 0) {
        SDLTest_Log("Harness input to repro failures:");
        for (testCounter = 0; testCounter < failedNumberOfTests; testCounter++) {
            SDLTest_Log(" --seed %s --filter %s", runSeed, failedTests[testCounter]->name);
        }
    }
    SDL_free((void *)failedTests);

    SDLTest_Log("Exit code: %d", runResult);
    return runResult;
}

/* vi: set ts=4 sw=4 expandtab: */
