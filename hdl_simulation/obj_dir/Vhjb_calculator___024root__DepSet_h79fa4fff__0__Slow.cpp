// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vhjb_calculator.h for the primary calling header

#include "Vhjb_calculator__pch.h"
#include "Vhjb_calculator___024root.h"

VL_ATTR_COLD void Vhjb_calculator___024root___eval_static(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_static\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
    vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0 = vlSelfRef.rst_n;
}

VL_ATTR_COLD void Vhjb_calculator___024root___eval_initial(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_initial\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vhjb_calculator___024root___eval_final(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_final\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vhjb_calculator___024root___eval_settle(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___eval_settle\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vhjb_calculator___024root___dump_triggers__act(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___dump_triggers__act\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge rst_n)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vhjb_calculator___024root___dump_triggers__nba(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___dump_triggers__nba\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge rst_n)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vhjb_calculator___024root___ctor_var_reset(Vhjb_calculator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vhjb_calculator___024root___ctor_var_reset\n"); );
    Vhjb_calculator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1638864771569018232ull);
    vlSelf->mid_price = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 1238515469326370040ull);
    vlSelf->inventory = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4593177094152973812ull);
    vlSelf->volatility = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 9570938037459815655ull);
    vlSelf->calculate_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16856200205308978235ull);
    vlSelf->optimal_bid = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 5123714553762186701ull);
    vlSelf->optimal_ask = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 14926302866087035526ull);
    vlSelf->calculation_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12411805887749505188ull);
    vlSelf->latency_cycles = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16032696644669046232ull);
    vlSelf->hjb_calculator__DOT__reservation_price = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 12047297646447673156ull);
    vlSelf->hjb_calculator__DOT__spread = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 17564236540106298882ull);
    vlSelf->hjb_calculator__DOT__cycle_counter = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3891055716330192009ull);
    vlSelf->hjb_calculator__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 6401177674272159377ull);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9526919608049418986ull);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14803524876191471008ull);
}
