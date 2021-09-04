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
    {
        msg_ = new char[256];
        memset(msg_, '\0', 256);
        strcpy(msg_, msg);
    }

    ~ScopeTimer()
    {
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
};

class ScopeTimerSec : public ScopeTimer<chrono::seconds>
{
public:
    ScopeTimerSec(const char* msg)
    :   ScopeTimer<chrono::seconds>(msg)
    {}

    void printMsg()
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

    void printMsg()
    {
        cout << "Timer - " << msg_ << " " << getCurrentDuration() << "m" << endl;
    }
};