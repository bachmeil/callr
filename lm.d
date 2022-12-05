module callr.lm;

import embedr.r;
alias rvec = RData!RVector;
alias rlist = RData!RList;

struct LM(T) {
  RData!RVector y;
  RData!T x;
  bool intercept = true;
  
  this(RData!RVector _y, RData!T _x) {
    y = _y;
    x = _x;
  } 

  LMFit ols() {
    string intTerm = "";
    if (!intercept) {
      intTerm = " -1";
    }
    auto fit = RData!RList(`lm(` ~ y.name ~ ` ~ ` ~ x.name ~ intTerm ~ `)`);
    return LMFit(fit);
  }
}

struct LMFit {
  RVector coefficients;
  RVector residuals;
  RVector effects;
  RVector fittedValues;
  int rank;
  int dfResidual;
  
  this(RData!RList fit) {
    coefficients = RVector(fit["coefficients"]);
    residuals = RVector(fit["residuals"]);
    effects = RVector(fit["effects"]);
    fittedValues = RVector(fit["fitted.values"]);
    rank = fit["rank"].scalar!int;
    dfResidual = fit["df.residual"].scalar!int;
  }
}
    

  
  
