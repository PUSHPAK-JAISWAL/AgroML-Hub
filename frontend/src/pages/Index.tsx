import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Sprout, Brain, TrendingUp, Shield } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card shadow-soft">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center">
            <Sprout className="h-8 w-8 text-primary mr-2" />
            <span className="text-2xl font-bold text-foreground">AgroML</span>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => navigate("/auth")}>
              Sign In
            </Button>
            <Button onClick={() => navigate("/auth")}>
              Get Started
            </Button>
          </div>
        </div>
      </header>

      <main>
        <section className="bg-gradient-hero text-primary-foreground py-20">
          <div className="container mx-auto px-4 text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              Agricultural Machine Learning Platform
            </h1>
            <p className="text-xl md:text-2xl mb-8 opacity-90 max-w-3xl mx-auto">
              Leverage advanced ML models for forest classification, seed analysis, and mushroom identification
            </p>
            <Button 
              size="lg" 
              variant="secondary"
              onClick={() => navigate("/auth")}
              className="text-lg px-8"
            >
              Start Predicting
            </Button>
          </div>
        </section>

        <section className="py-16 bg-background">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center text-foreground mb-12">
              Powerful Features
            </h2>
            <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent mb-4">
                  <Brain className="h-8 w-8 text-accent-foreground" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-foreground">
                  Multiple Models
                </h3>
                <p className="text-muted-foreground">
                  Compare predictions from Random Forest, XGBoost, TensorFlow, and quantized models
                </p>
              </div>

              <div className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent mb-4">
                  <TrendingUp className="h-8 w-8 text-accent-foreground" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-foreground">
                  Real-time Analysis
                </h3>
                <p className="text-muted-foreground">
                  Get instant predictions with confidence scores and detailed probability distributions
                </p>
              </div>

              <div className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent mb-4">
                  <Shield className="h-8 w-8 text-accent-foreground" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-foreground">
                  Prediction History
                </h3>
                <p className="text-muted-foreground">
                  Track all your predictions with automatic email notifications and historical data
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 bg-muted">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center text-foreground mb-12">
              Available Models
            </h2>
            <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
              <div className="bg-card p-6 rounded-lg shadow-soft">
                <h3 className="text-xl font-semibold mb-3 text-card-foreground">
                  Forest Covertype
                </h3>
                <p className="text-muted-foreground text-sm">
                  Predict forest cover types using cartographic variables including elevation, slope, and distance metrics.
                </p>
              </div>

              <div className="bg-card p-6 rounded-lg shadow-soft">
                <h3 className="text-xl font-semibold mb-3 text-card-foreground">
                  Wheat Seeds
                </h3>
                <p className="text-muted-foreground text-sm">
                  Classify wheat kernel varieties based on geometric measurements like area, perimeter, and compactness.
                </p>
              </div>

              <div className="bg-card p-6 rounded-lg shadow-soft">
                <h3 className="text-xl font-semibold mb-3 text-card-foreground">
                  Mushroom Classification
                </h3>
                <p className="text-muted-foreground text-sm">
                  Identify mushroom characteristics using 22 categorical features from cap to habitat information.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 bg-background text-center">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-foreground mb-6">
              Ready to Get Started?
            </h2>
            <p className="text-muted-foreground text-lg mb-8 max-w-2xl mx-auto">
              Create your account and start making predictions with state-of-the-art machine learning models
            </p>
            <Button size="lg" onClick={() => navigate("/auth")}>
              Sign Up Now
            </Button>
          </div>
        </section>
      </main>

      <footer className="border-t border-border bg-card py-8">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>&copy; 2025 AgroML. Agricultural Machine Learning Platform.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
