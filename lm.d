module callr.lm;

import std.sumtype;
import embedr.r;

alias r_vec = RData!RVector;
alias r_mat = RData!RMatrix;
alias r_list = RData!RList;
alias r_array = RData!RArrayInsideR;

struct LMStruct(T) {
  r_vec y;
  T x;
  bool intercept = true;

  LMFit ols() {
    string intTerm = "";
    if (!intercept) {
      intTerm = " -1";
    }
    auto fit = r_list(`lm(` ~ y.name ~ ` ~ ` ~ x.name ~ intTerm ~ `)`);
    return LMFit(fit);
  }
}

LMStruct!T LM(T)(r_vec y, T x) {
  return LMStruct!T(y, x);
}

struct LMFit {
  RVector coefficients;
  RVector residuals;
  RVector effects;
  RVector fittedValues;
  int rank;
  int dfResidual;
  
  this(r_list fit) {
    coefficients = RVector(fit["coefficients"]);
    residuals = RVector(fit["residuals"]);
    effects = RVector(fit["effects"]);
    fittedValues = RVector(fit["fitted.values"]);
    rank = fit["rank"].scalar!int;
    dfResidual = fit["df.residual"].scalar!int;
  }
}
    

  
  
