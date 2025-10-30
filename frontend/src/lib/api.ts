const API_BASE_URL = "http://localhost:8080";

export interface AuthResponse {
  token: string;
  tokenType: string;
}

export interface User {
  id: string | { timestamp: number; date: string };
  email: string;
  roles: string[];
  createdAt?: string;
  updatedAt?: string;
}

// Helper to convert MongoDB ObjectId to string
export const getUserId = (user: User): string => {
  if (typeof user.id === 'string') return user.id;
  return user.id.timestamp.toString();
};

export interface PredictionRequest {
  modelPath: string;
  sample: Record<string, any>;
}

export interface PredictionHistory {
  userId: string;
  endpoint: string;
  input: any;
  response: any;
  createdAt: string;
}

class ApiClient {
  private token: string | null = null;

  setToken(token: string | null) {
    this.token = token;
    if (token) {
      localStorage.setItem("agroml_token", token);
    } else {
      localStorage.removeItem("agroml_token");
    }
  }

  getToken(): string | null {
    if (!this.token) {
      this.token = localStorage.getItem("agroml_token");
    }
    return this.token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
      ...options.headers,
    };

    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorMessage = `Request failed with status ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorData.message || errorData.detail || errorMessage;
      } catch {
        const errorText = await response.text();
        if (errorText) errorMessage = errorText;
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }

  async register(email: string, password: string): Promise<User> {
    return this.request<User>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  async login(email: string, password: string): Promise<AuthResponse> {
    const response = await this.request<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    this.setToken(response.token);
    return response;
  }

  logout() {
    this.setToken(null);
  }

  async getMe(): Promise<{ user: User; isAdmin: boolean }> {
    return this.request<{ user: User; isAdmin: boolean }>("/auth/me");
  }

  async isAdmin(): Promise<{ isAdmin: boolean }> {
    return this.request<{ isAdmin: boolean }>("/auth/is-admin");
  }

  async predict(request: PredictionRequest): Promise<any> {
    return this.request<any>("/predictions", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async getPredictionHistory(): Promise<PredictionHistory[]> {
    return this.request<PredictionHistory[]>("/predictions/history");
  }

  async getUsers(): Promise<User[]> {
    return this.request<User[]>("/admin/users");
  }

  async createUser(user: { email: string; password: string; roles: string[] }): Promise<User> {
    return this.request<User>("/admin/users", {
      method: "POST",
      body: JSON.stringify(user),
    });
  }

  async updateUser(id: string, user: { email: string; password?: string; roles: string[] }): Promise<User> {
    return this.request<User>(`/admin/users/${id}`, {
      method: "PUT",
      body: JSON.stringify(user),
    });
  }

  async deleteUser(id: string): Promise<void> {
    await this.request<void>(`/admin/users/${id}`, {
      method: "DELETE",
    });
  }
}

export const api = new ApiClient();
