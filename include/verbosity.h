/*
 *  verbosity.h
 *  
 *
 *  Created by Nicolas Garnier on 2020/03/05.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  
 *  2022-12-14 : new functions to print output
 */
#ifndef _LIBRARY_VERBOSITY_H
#define _LIBRARY_VERBOSITY_H

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

/********************************************************************************************************************/
// global variables that changes the behavior of the library
extern int lib_verbosity;               // defined in library_commons.c
extern int lib_warning_level;           // defined in library_commons.c

/********************************************************************************************************************/
void ANN_set_verbosity(int level);      // to set the verbosity

/********************************************************************************************************************/
// shortcut functions to enhanced output of informations:
void print_warning(char *func_name, char *message);
int  print_error  (char *func_name, char *message);
void print_info   (char *func_name, char *message);

#endif


