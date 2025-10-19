// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vhjb_calculator.h for the primary calling header

#include "Vhjb_calculator__pch.h"
#include "Vhjb_calculator___024root.h"

void Vhjb_calculator___024root___eval_act(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_act\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

void Vhjb_calculator___024root___nba_sequent__TOP__0(Vhjb_calculator___024root* vlSelf);

void Vhjb_calculator___024root___eval_nba(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_nba\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vhjb_calculator___024root___nba_sequent__TOP__0(vlSelf);
    }
}

VL_INLINE_OPT void Vhjb_calculator___024root___nba_sequent__TOP__0(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___nba_sequent__TOP__0\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*2:0*/ __Vdly__hjb_calculator__DOT__state;
    __Vdly__hjb_calculator__DOT__state = 0;
    IData/*31:0*/ __Vdly__hjb_calculator__DOT__cycle_counter;
    __Vdly__hjb_calculator__DOT__cycle_counter = 0;
    // Body
    __Vdly__hjb_calculator__DOT__state = vlSelfRef.hjb_calculator__DOT__state;
    __Vdly__hjb_calculator__DOT__cycle_counter = vlSelfRef.hjb_calculator__DOT__cycle_counter;
    if (vlSelfRef.rst_n) {
        if ((4U & (IData)(vlSelfRef.hjb_calculator__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.hjb_calculator__DOT__state))) {
                __Vdly__hjb_calculator__DOT__state = 0U;
            } else if ((1U & (IData)(vlSelfRef.hjb_calculator__DOT__state))) {
                __Vdly__hjb_calculator__DOT__state = 0U;
            } else {
                vlSelfRef.calculation_done = 1U;
                vlSelfRef.latency_cycles = vlSelfRef.hjb_calculator__DOT__cycle_counter;
                if ((1U & (~ (IData)(vlSelfRef.calculate_en)))) {
                    __Vdly__hjb_calculator__DOT__state = 0U;
                }
            }
        } else if ((2U & (IData)(vlSelfRef.hjb_calculator__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.hjb_calculator__DOT__state))) {
                __Vdly__hjb_calculator__DOT__cycle_counter 
                    = ((IData)(1U) + vlSelfRef.hjb_calculator__DOT__cycle_counter);
                vlSelfRef.optimal_bid = (vlSelfRef.hjb_calculator__DOT__reservation_price 
                                         - VL_SHIFTR_QQI(64,64,32, vlSelfRef.hjb_calculator__DOT__spread, 1U));
                vlSelfRef.optimal_ask = (vlSelfRef.hjb_calculator__DOT__reservation_price 
                                         + VL_SHIFTR_QQI(64,64,32, vlSelfRef.hjb_calculator__DOT__spread, 1U));
                __Vdly__hjb_calculator__DOT__state = 4U;
            } else {
                __Vdly__hjb_calculator__DOT__cycle_counter 
                    = ((IData)(1U) + vlSelfRef.hjb_calculator__DOT__cycle_counter);
                vlSelfRef.hjb_calculator__DOT__spread 
                    = VL_SHIFTR_QQI(64,64,32, vlSelfRef.mid_price, 7U);
                __Vdly__hjb_calculator__DOT__state = 3U;
            }
        } else if ((1U & (IData)(vlSelfRef.hjb_calculator__DOT__state))) {
            __Vdly__hjb_calculator__DOT__cycle_counter 
                = ((IData)(1U) + vlSelfRef.hjb_calculator__DOT__cycle_counter);
            vlSelfRef.hjb_calculator__DOT__reservation_price 
                = (vlSelfRef.mid_price - VL_SHIFTL_QQI(64,64,32, (QData)((IData)(vlSelfRef.inventory)), 0xaU));
            __Vdly__hjb_calculator__DOT__state = 2U;
        } else {
            __Vdly__hjb_calculator__DOT__cycle_counter = 0U;
            vlSelfRef.calculation_done = 0U;
            if (vlSelfRef.calculate_en) {
                __Vdly__hjb_calculator__DOT__cycle_counter 
                    = ((IData)(1U) + vlSelfRef.hjb_calculator__DOT__cycle_counter);
                __Vdly__hjb_calculator__DOT__state = 1U;
            }
        }
    } else {
        __Vdly__hjb_calculator__DOT__cycle_counter = 0U;
        __Vdly__hjb_calculator__DOT__state = 0U;
        vlSelfRef.optimal_bid = 0ULL;
        vlSelfRef.optimal_ask = 0ULL;
        vlSelfRef.calculation_done = 0U;
        vlSelfRef.latency_cycles = 0U;
    }
    vlSelfRef.hjb_calculator__DOT__state = __Vdly__hjb_calculator__DOT__state;
    vlSelfRef.hjb_calculator__DOT__cycle_counter = __Vdly__hjb_calculator__DOT__cycle_counter;
}

void Vhjb_calculator___024root___eval_triggers__act(Vhjb_calculator___024root* vlSelf);

bool Vhjb_calculator___024root___eval_phase__act(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_phase__act\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vhjb_calculator___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vhjb_calculator___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vhjb_calculator___024root___eval_phase__nba(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_phase__nba\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vhjb_calculator___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vhjb_calculator___024root___dump_triggers__nba(Vhjb_calculator___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vhjb_calculator___024root___dump_triggers__act(Vhjb_calculator___024root* vlSelf);
#endif  // VL_DEBUG

void Vhjb_calculator___024root___eval(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY(((0x64U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vhjb_calculator___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("rtl/hjb_calculator.v", 2, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vhjb_calculator___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("rtl/hjb_calculator.v", 2, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vhjb_calculator___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vhjb_calculator___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vhjb_calculator___024root___eval_debug_assertions(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_debug_assertions\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((vlSelfRef.clk & 0xfeU)))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY(((vlSelfRef.rst_n & 0xfeU)))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY(((vlSelfRef.calculate_en & 0xfeU)))) {
        Verilated::overWidthError("calculate_en");}
}
#endif  // VL_DEBUG
