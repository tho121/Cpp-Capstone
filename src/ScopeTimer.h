#include <string>
#include <chrono>
#include <iostream>

using namespace std;

template<typename T>
class ScopeTimer
{
public:
    
    ScopeTimer(const char* msg)
    :   startPoint_(chrono::high_resolution_clock::now())
    ,   finalMsgPrinted_(false)
    {
        msg_ = new char[256];
        memset(msg_, '\0', 256);
        strcpy(msg_, msg);
    }

    virtual ~ScopeTimer()
    {
        if(!finalMsgPrinted_)
            printMsg();

        delete [] msg_;
    }

    ScopeTimer(const ScopeTimer& other)
    {
        strcpy(msg_, other.msg_);
        startPoint_ = other.startPoint_;
    }

    ScopeTimer& operator=(const ScopeTimer& other)
    {
        strcpy(msg_, other.msg_);
        startPoint_ = other.startPoint_;

        return *this;
    }

    ScopeTimer(ScopeTimer&& other)
    {
        msg_ = other.msg_;
        other.msg_ = nullptr;

        startPoint_ = std::move(other.startPoint_);
    }

    ScopeTimer& operator=(ScopeTimer&& other)
    {
        msg_ = other.msg_;
        other.msg_ = nullptr;

        startPoint_ = std::move(other.startPoint_);

        return *this;
    }

    int getCurrentDuration()
    {
        return chrono::duration_cast<T>(chrono::high_resolution_clock::now() - startPoint_).count();
    }

    virtual void printMsg()
    {
        cout << "Timer - " << msg_ << " " << getCurrentDuration() << endl;
    }

protected:
    char* msg_;
    chrono::time_point<std::chrono::high_resolution_clock> startPoint_;
    bool finalMsgPrinted_;
};

class ScopeTimerSec : public ScopeTimer<chrono::seconds>
{
public:
    ScopeTimerSec(const char* msg)
    :   ScopeTimer<chrono::seconds>(msg)
    {}

    virtual ~ScopeTimerSec()
    {
        printMsg();

        finalMsgPrinted_ = true;
    }

    virtual void printMsg()
    {
        cout << "Timer - " << msg_ << " " << getCurrentDuration() << "s" << endl;
    }
};

class ScopeTimerMin : public ScopeTimer<chrono::minutes>
{
public:
    ScopeTimerMin(const char* msg)
    :   ScopeTimer<chrono::minutes>(msg)
    {}

    virtual ~ScopeTimerMin()
    {
        printMsg();

        finalMsgPrinted_ = true;
    }

    virtual void printMsg()
    {
        cout << "Timer - " << msg_ << " " << getCurrentDuration() << "m" << endl;
    }
};