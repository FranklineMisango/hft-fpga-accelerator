// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vhjb_calculator__pch.h"

//============================================================
// Constructors

Vhjb_calculator::Vhjb_calculator(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vhjb_calculator__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , calculate_en{vlSymsp->TOP.calculate_en}
    , calculation_done{vlSymsp->TOP.calculation_done}
    , inventory{vlSymsp->TOP.inventory}
    , latency_cycles{vlSymsp->TOP.latency_cycles}
    , mid_price{vlSymsp->TOP.mid_price}
    , volatility{vlSymsp->TOP.volatility}
    , optimal_bid{vlSymsp->TOP.optimal_bid}
    , optimal_ask{vlSymsp->TOP.optimal_ask}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vhjb_calculator::Vhjb_calculator(const char* _vcname__)
    : Vhjb_calculator(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vhjb_calculator::~Vhjb_calculator() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vhjb_calculator___024root___eval_debug_assertions(Vhjb_calculator___024root* vlSelf);
#endif  // VL_DEBUG
void Vhjb_calculator___024root___eval_static(Vhjb_calculator___024root* vlSelf);
void Vhjb_calculator___024root___eval_initial(Vhjb_calculator___024root* vlSelf);
void Vhjb_calculator___024root___eval_settle(Vhjb_calculator___024root* vlSelf);
void Vhjb_calculator___024root___eval(Vhjb_calculator___024root* vlSelf);

void Vhjb_calculator::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vhjb_calculator::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vhjb_calculator___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vhjb_calculator___024root___eval_static(&(vlSymsp->TOP));
        Vhjb_calculator___024root___eval_initial(&(vlSymsp->TOP));
        Vhjb_calculator___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vhjb_calculator___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vhjb_calculator::eventsPending() { return false; }

uint64_t Vhjb_calculator::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vhjb_calculator::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vhjb_calculator___024root___eval_final(Vhjb_calculator___024root* vlSelf);

VL_ATTR_COLD void Vhjb_calculator::final() {
    Vhjb_calculator___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vhjb_calculator::hierName() const { return vlSymsp->name(); }
const char* Vhjb_calculator::modelName() const { return "Vhjb_calculator"; }
unsigned Vhjb_calculator::threads() const { return 1; }
void Vhjb_calculator::prepareClone() const { contextp()->prepareClone(); }
void Vhjb_calculator::atClone() const {
    contextp()->threadPoolpOnClone();
}
