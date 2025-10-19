// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VHJB_CALCULATOR__SYMS_H_
#define VERILATED_VHJB_CALCULATOR__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vhjb_calculator.h"

// INCLUDE MODULE CLASSES
#include "Vhjb_calculator___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vhjb_calculator__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vhjb_calculator* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vhjb_calculator___024root      TOP;

    // CONSTRUCTORS
    Vhjb_calculator__Syms(VerilatedContext* contextp, const char* namep, Vhjb_calculator* modelp);
    ~Vhjb_calculator__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
