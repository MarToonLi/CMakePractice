#pragma once

#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <Windows.h>   
#include <iostream>    
#include <string> 
using namespace std;


class SLogger
{
    string GetProgramDir()
    {
        char exeFullPath[MAX_PATH]; // Full path 
        string strPath = "";

        GetModuleFileName(NULL, exeFullPath, MAX_PATH);
        strPath = (string)exeFullPath;    // Get full path of the file 

        int pos = strPath.find_last_of('\\', strPath.length());
        return strPath.substr(0, pos);  // Return the directory without the file name 
    }

public:
    static SLogger& getInstance()
    {
        static SLogger log;
        return log;
    }

    virtual ~SLogger()
    {
        spdlog::drop(LOGGER_NAME_FILE.c_str());
        spdlog::drop(LOGGER_NAME_CONSOLE.c_str());
        asyncLogger = nullptr;
        consoleLogger = nullptr;
    }

    SLogger()
    {
        spdlog::set_pattern("[%C:%m:%d-%H:%M:%S.%e][%s:%#:%!][tid:%t][%^%l%$] %v");
        spdlog::set_level(spdlog::level::debug);  // Set global log level to debug
        //    spdlog::init_thread_pool(8 * 1024, 1); // queue with 8k items and 1 backing thread.
        spdlog::flush_every(std::chrono::seconds(5));

        asyncLogger = spdlog::get(LOGGER_NAME_FILE);
        if (!asyncLogger)
        {
            asyncLogger = spdlog::daily_logger_format_mt<spdlog::async_factory>(LOGGER_NAME_FILE,
                GetProgramDir() + "/logs/%Y_%m_%d/algo_logger_%H_%M_%S_%e.txt");
            asyncLogger->flush_on(spdlog::level::trace);
        }

        consoleLogger = spdlog::get(LOGGER_NAME_CONSOLE);

        if (!consoleLogger)
        {
            consoleLogger = spdlog::stdout_color_mt(LOGGER_NAME_CONSOLE);
        }
    }
    void SetLogger(int _level)
    {
        spdlog::set_level((spdlog::level::level_enum)_level);  // Set global log level to debug
    }

    std::shared_ptr<spdlog::logger> asyncLogger;
    std::shared_ptr<spdlog::logger> consoleLogger;

private:
    std::string LOGGER_NAME_FILE = "daily_async_logger";
    std::string LOGGER_NAME_CONSOLE = "console";
};

#define LOGSET(_logger)                                                                            \
    {                                                                                              \
        if (SLogger::getInstance().consoleLogger)                                                   \
            SLogger::getInstance().SetLogger(_logger);                                              \
    }
#define LOGT(...)                                                                                  \
    {                                                                                              \
        if (SLogger::getInstance().consoleLogger)                                                   \
            SLogger::getInstance().consoleLogger->log(                                              \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::trace,\
                __VA_ARGS__);                                                                      \
        if (SLogger::getInstance().asyncLogger)                                                     \
            SLogger::getInstance().asyncLogger->log(                                                \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::trace,\
                __VA_ARGS__);                                                                      \
    }
#define LOGI(...)                                                                                  \
    {                                                                                              \
        if (SLogger::getInstance().consoleLogger)                                                   \
            SLogger::getInstance().consoleLogger->log(                                              \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::info, \
                __VA_ARGS__);                                                                      \
        if (SLogger::getInstance().asyncLogger)                                                     \
            SLogger::getInstance().asyncLogger->log(                                                \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::info, \
                __VA_ARGS__);                                                                      \
    }
#define LOGD(...)                                                 \
    {                                                             \
        if (SLogger::getInstance().consoleLogger)                  \
            SLogger::getInstance().consoleLogger->log(             \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, \
                spdlog::level::level_enum::debug, __VA_ARGS__);   \
        if (SLogger::getInstance().asyncLogger)                    \
            SLogger::getInstance().asyncLogger->log(               \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, \
                spdlog::level::level_enum::debug, __VA_ARGS__);   \
    }
#define LOGW(...)                                                                                  \
    {                                                                                              \
        if (SLogger::getInstance().consoleLogger)                                                   \
            SLogger::getInstance().consoleLogger->log(                                              \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::warn, \
                __VA_ARGS__);                                                                      \
        if (SLogger::getInstance().asyncLogger)                                                     \
            SLogger::getInstance().asyncLogger->log(                                                \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::warn, \
                __VA_ARGS__);                                                                      \
    }
#define LOGE(...)                                                                                 \
    {                                                                                             \
        if (SLogger::getInstance().consoleLogger)                                                  \
            SLogger::getInstance().consoleLogger->log(                                             \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::err, \
                __VA_ARGS__);                                                                     \
        if (SLogger::getInstance().asyncLogger)                                                    \
            SLogger::getInstance().asyncLogger->log(                                               \
                spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::level_enum::err, \
                __VA_ARGS__);                                                                     \
    }
