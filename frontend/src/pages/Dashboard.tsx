import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Trees, Wheat, Circle, History, LogOut, Shield } from "lucide-react";
import { toast } from "sonner";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { logout } from "@/store/authSlice";

const Dashboard = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const isAdmin = useAppSelector((state) => state.auth.isAdmin);

  const handleLogout = () => {
    dispatch(logout());
    toast.success("Logged out successfully");
    navigate("/auth");
  };

  const models = [
    {
      id: "forest",
      name: "Forest Covertype",
      description: "Predict forest cover type based on cartographic variables",
      icon: Trees,
      path: "/predict/forest",
      color: "text-primary",
    },
    {
      id: "seeds",
      name: "Wheat Seeds",
      description: "Classify wheat kernel varieties using geometric properties",
      icon: Wheat,
      path: "/predict/seeds",
      color: "text-secondary",
    },
    {
      id: "mushrooms",
      name: "Mushroom Classification",
      description: "Identify mushroom edibility from physical characteristics",
      icon: Circle,
      path: "/predict/mushrooms",
      color: "text-accent-foreground",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card shadow-soft">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-foreground">AgroML Dashboard</h1>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => navigate("/history")}>
              <History className="mr-2 h-4 w-4" />
              History
            </Button>
            {isAdmin && (
              <Button variant="outline" onClick={() => navigate("/admin")}>
                <Shield className="mr-2 h-4 w-4" />
                Admin
              </Button>
            )}
            <Button variant="ghost" onClick={handleLogout}>
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-foreground mb-4">
            Choose Your Model
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Select a machine learning model to make predictions on agricultural data
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {models.map((model) => {
            const Icon = model.icon;
            return (
              <Card
                key={model.id}
                className="p-6 hover:shadow-medium transition-all cursor-pointer bg-gradient-card"
                onClick={() => navigate(model.path)}
              >
                <Icon className={`h-12 w-12 mb-4 ${model.color}`} />
                <h3 className="text-xl font-semibold mb-2 text-card-foreground">
                  {model.name}
                </h3>
                <p className="text-muted-foreground text-sm mb-4">
                  {model.description}
                </p>
                <Button className="w-full">
                  Start Prediction
                </Button>
              </Card>
            );
          })}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
