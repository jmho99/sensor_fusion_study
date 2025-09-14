#ifndef SERVE_KEYBOARD_HPP
#define SERVE_KEYBOARD_HPP

#include <unistd.h>
#include <sys/select.h>
#include <sys/time.h>

namespace jmh_utils
{
    bool keyboardAvailable();
}

#endif