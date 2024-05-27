// verbosity.c
//
// routines for printing warnings and errors
// 2022-12-14, N.B.G.

#include <stdio.h>
#include "verbosity.h"

void print_warning(char *func_name, char *message)
{   printf("[%s] " ANSI_COLOR_BLUE "warning " ANSI_COLOR_RESET "%s\n", func_name, message);
    return;
}

int print_error(char *func_name, char *message)
{   printf("[%s] " ANSI_COLOR_RED "error " ANSI_COLOR_RESET "%s\n", func_name, message);
    return(-1);
}

void print_info(char *func_name, char *message)
{   printf("[%s] " ANSI_COLOR_YELLOW "info " ANSI_COLOR_RESET "%s\n", func_name, message);
    return;
}


