QT += core
QT -= gui

TARGET = mlp-standard
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

HEADERS += \
    random_generator.h \
    recursive_accumulator.h \
    input_vectors.h \
    target_size.h \
    #target_lmssim.h \
    #target_time.h

QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS_RELEASE += -O3 -mavx -mavx2

#QMAKE_LFLAGS += -static
