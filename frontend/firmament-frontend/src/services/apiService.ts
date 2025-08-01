import NetworkUtils, { NetworkError } from '../utils/networkUtils';
import config from '../config';
import AIService from './aiService';

// Auth-related interfaces
export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    name: string;
    picture?: string;
  };
}

export interface UserInfo {
  id: string;
  email: string;
  name: string;
  picture?: string;
}

// Define proper types for API responses
export interface UploadResponse {
  job_id: string;
  filename: string;
  filetype: string;
  message: string;
  text?: string;
  transcription_file?: string;
}

export interface ProcessChunksResponse {
  num_chunks: number;
  total_words: number;
}

export interface TopicResponse {
  num_chunks: number;
  num_topics: number;
  total_tokens_used: number;
  segments?: Array<{ position: string; text: string }>;
  topics: Record<string, Topic>;
}

export interface TopicSegment {
  position: string;
  text: string;
  topic_id?: number;
  cluster_id?: number;
}

export interface Topic {
  heading: string;
  examples: string[];
  chunks: TopicSegment[];
  bullet_points?: string[];
  cluster_id?: number;
  size?: number;
  representative_docs?: string[];
  concepts?: string[];
  summary?: string;
  keywords?: string[];
  segment_positions?: string[];
  stats?: {
    num_chunks: number;
    min_size: number;
    mean_size: number;
    max_size: number;
  };
  bullet_expansions?: {
    [bulletKey: string]: {
      original_bullet?: string;
      expanded_bullets?: string[];
      topic_heading?: string;
      chunks_used?: number;
      layer?: number;
      timestamp?: string;
      sub_expansions?: {
        [subBulletKey: string]: {
          original_bullet?: string;
          expanded_bullets?: string[];
          layer?: number;
          topic_heading?: string;
          chunks_used?: number;
          timestamp?: string;
        };
      };
    };
  };
}

export interface ExpandClusterResponse {
  message: string;
  cluster: Topic;
}

export interface BulletPointData {
  bullet_point: string;
  chunks: string[];
  topic_heading: string;
  filename: string;
  topic_id: string;
  layer?: number;
  other_bullets?: string[];
  parent_bullet?: string;
}

export interface BulletPointExpandResponse {
  original_bullet: string;
  expanded_bullets: string[];
  topic_heading: string;
  chunks_used: number;
  layer: number;
  error?: string;
}

export interface ProgressData {
  stage: string;
  current?: number;
  total?: number;
  result?: unknown;
  error?: string;
}

export interface ExpandClusterRequest {
  filename: string;
  cluster_id: string | number;
}

class ApiService {
  private static instance: ApiService;
  private networkUtils: NetworkUtils;

  private constructor() {
    this.networkUtils = NetworkUtils.getInstance(config.getApiBaseUrl());
  }

  public static getInstance(): ApiService {
    if (!ApiService.instance) {
      ApiService.instance = new ApiService();
    }
    return ApiService.instance;
  }

  /**
   * Enhanced fetch wrapper with automatic retry and better error handling
   */
  private async enhancedFetch(
    endpoint: string,
    options: RequestInit = {},
    retryOptions?: { maxRetries?: number; baseDelay?: number }
  ): Promise<Response> {
    const url = config.getApiUrl(endpoint);
    
    return this.networkUtils.fetchWithRetry(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    }, retryOptions);
  }

  /**
   * Handle response and extract JSON with proper error handling
   */
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorText = await response.text();
      let errorData;
      
      try {
        errorData = JSON.parse(errorText);
      } catch {
        errorData = { error: errorText || `HTTP ${response.status}: ${response.statusText}` };
      }

      throw new Error(errorData.error || errorData.message || `Request failed with status ${response.status}`);
    }

    return response.json();
  }

  /**
   * Add authorization header to request options
   */
  private addAuthHeader(options: RequestInit = {}): RequestInit {
    const token = localStorage.getItem('studymate_token');
    if (token) {
      return {
        ...options,
        headers: {
          ...options.headers,
          'Authorization': `Bearer ${token}`,
        },
      };
    }
    return options;
  }

  /**
   * Authentication: Sign in with Google
   */
  public async signInWithGoogle(googleToken: string): Promise<AuthResponse> {
    try {
      const response = await this.enhancedFetch('auth/google', {
        method: 'POST',
        body: JSON.stringify({ token: googleToken }),
      });

      return this.handleResponse<AuthResponse>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Google sign-in failed. Please try again.');
    }
  }

  /**
   * Authentication: Get current user info
   */
  public async getCurrentUser(): Promise<UserInfo> {
    try {
      const response = await this.enhancedFetch('auth/me', this.addAuthHeader({
        method: 'GET',
      }));

      return this.handleResponse<UserInfo>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Failed to get user info. Please try again.');
    }
  }

  /**
   * Authentication: Sign out
   */
  public async signOut(): Promise<{ message: string }> {
    try {
      const response = await this.enhancedFetch('auth/logout', this.addAuthHeader({
        method: 'POST',
      }));

      return this.handleResponse<{ message: string }>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Sign out failed. Please try again.');
    }
  }

  /**
   * Upload file
   */
  public async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use direct fetch for file uploads to avoid Content-Type issues
      const url = config.getApiUrl('upload');
      
      // Add authorization header for authenticated uploads
      const headers: Record<string, string> = {};
      const token = localStorage.getItem('studymate_token');
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      
      const response = await this.networkUtils.fetchWithRetry(url, {
        method: 'POST',
        body: formData,
        headers,
        // Don't set Content-Type - let the browser handle multipart/form-data
      }, {
        maxRetries: 2, // Fewer retries for uploads
        baseDelay: 2000 // Longer delay between retries
      });

      return this.handleResponse<UploadResponse>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Upload failed. Please try again.');
    }
  }

  /**
   * Process text chunks
   */
  public async processChunks(
    text: string,
    filename: string
  ): Promise<ProcessChunksResponse> {
    try {
      const response = await this.enhancedFetch('process-chunks', {
        method: 'POST',
        body: JSON.stringify({ text, filename }),
      });

      return this.handleResponse<ProcessChunksResponse>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Failed to process chunks. Please try again.');
    }
  }

  /**
   * Generate topic headings
   */
  public async generateHeadings(filename: string): Promise<TopicResponse> {
    try {
      const response = await this.enhancedFetch('generate-headings', {
        method: 'POST',
        body: JSON.stringify({ filename }),
      });

      return this.handleResponse<TopicResponse>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Failed to generate headings. Please try again.');
    }
  }

  /**
   * Expand cluster
   */
  public async expandCluster(data: ExpandClusterRequest): Promise<ExpandClusterResponse> {
    try {
      const response = await this.enhancedFetch('expand-cluster', {
        method: 'POST',
        body: JSON.stringify(data),
      });

      return this.handleResponse<ExpandClusterResponse>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Failed to expand cluster. Please try again.');
    }
  }

  /**
   * Expand bullet point
   */
  public async expandBulletPoint(data: BulletPointData): Promise<BulletPointExpandResponse> {
    try {
      const response = await this.enhancedFetch('expand-bullet-point', {
        method: 'POST',
        body: JSON.stringify(data),
      });

      return this.handleResponse<BulletPointExpandResponse>(response);
    } catch (error) {
      if (error instanceof Error) {
        const networkError = error as NetworkError;
        throw new Error(JSON.stringify(this.networkUtils.getErrorMessage(networkError)));
      }
      throw new Error('Failed to expand bullet point. Please try again.');
    }
  }

  /**
   * Generate topics using AIService
   */
  public async generateTopics(text: string): Promise<TopicResponse> {
    try {
      const aiTopics = await AIService.generateTopics(text);
      const topicResponse: TopicResponse = {
        num_chunks: aiTopics.topics.length,
        num_topics: aiTopics.topics.length,
        total_tokens_used: 0, // Placeholder, adjust as needed
        topics: aiTopics.topics.reduce((acc: Record<string, Topic>, topic: { title: string; bullets: string[] }) => {
          acc[topic.title] = {
            heading: topic.title,
            bullet_points: topic.bullets,
            chunks: [],
            examples: [], // Placeholder for examples
          };
          return acc;
        }, {}),
      };
      return topicResponse;
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to generate topics: ${error.message}`);
      }
      throw new Error('Failed to generate topics due to an unknown error.');
    }
  }

  /**
   * Generate bullet points using AIService
   */
  public async generateBullets(topicText: string): Promise<string[]> {
    try {
      const bullets = await AIService.generateBullets(topicText);
      return bullets;
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to generate bullets: ${error.message}`);
      }
      throw new Error('Failed to generate bullets due to an unknown error.');
    }
  }

  /**
   * Create Server-Sent Events connection for progress tracking
   */
  public createProgressEventSource(
    jobId: string,
    onMessage: (data: ProgressData) => void,
    onError?: (error: Error) => void,
    onClose?: () => void
  ): EventSource {
    const url = config.getApiUrl(`progress/${jobId}`);
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ProgressData;
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse SSE data:', error);
        onError?.(new Error('Failed to parse server response'));
      }
    };

    eventSource.onerror = (event) => {
      console.error('SSE error:', event);
      const error = new Error('Connection to server lost. Please try refreshing the page.');
      onError?.(error);
      
      // Auto-close on error
      eventSource.close();
      onClose?.();
    };

    // Auto-cleanup after 10 minutes to prevent memory leaks
    setTimeout(() => {
      if (eventSource.readyState !== EventSource.CLOSED) {
        eventSource.close();
        onClose?.();
      }
    }, 10 * 60 * 1000);

    return eventSource;
  }

  /**
   * Check if the service is healthy
   */
  public async checkHealth(): Promise<boolean> {
    try {
      const healthStatus = await this.networkUtils.checkBackendHealth();
      return healthStatus.isOnline;
    } catch {
      return false;
    }
  }

  /**
   * Get current network status
   */
  public getNetworkStatus() {
    return this.networkUtils.getHealthStatus();
  }
}

export default ApiService.getInstance();
