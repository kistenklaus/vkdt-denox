#!/usr/bin/env sh

######################################################################
# @author      : kistenklaus (karlsasssie@gmail.com)
# @file        : install
# @created     : Freitag Jan 16, 2026 20:07:45 CET
#
# @description : 
######################################################################


git archive --format=tar.gz --prefix=vkdt-denox-0.0.1/ -o vkdt-denox-0.0.1.tar.gz HEAD
makepkg -si
rm vkdt-denox-0.0.1.tar.gz
rm vkdt-denox-0.0.1-1-x86_64.pkg.tar.zst
