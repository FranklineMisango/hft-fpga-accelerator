// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vhjb_calculator.h for the primary calling header

#ifndef VERILATED_VHJB_CALCULATOR___024ROOT_H_
#define VERILATED_VHJB_CALCULATOR___024ROOT_H_  // guard

#include "verilated.h"


class Vhjb_calculator__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vhjb_calculator___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(calculate_en,0,0);
    VL_OUT8(calculation_done,0,0);
    CData/*2:0*/ hjb_calculator__DOT__state;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VL_IN(inventory,31,0);
    VL_OUT(latency_cycles,31,0);
    IData/*31:0*/ hjb_calculator__DOT__cycle_counter;
    IData/*31:0*/ __VactIterCount;
    VL_IN64(mid_price,63,0);
    VL_IN64(volatility,63,0);
    VL_OUT64(optimal_bid,63,0);
    VL_OUT64(optimal_ask,63,0);
    QData/*63:0*/ hjb_calculator__DOT__reservation_price;
    QData/*63:0*/ hjb_calculator__DOT__spread;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vhjb_calculator__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vhjb_calculator___024root(Vhjb_calculator__Syms* symsp, const char* v__name);
    ~Vhjb_calculator___024root();
    VL_UNCOPYABLE(Vhjb_calculator___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
