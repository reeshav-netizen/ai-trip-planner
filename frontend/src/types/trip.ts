export interface TripRequest {
  destination: string;
  duration: string;
  budget?: string;
  interests?: string;
  travel_style?: string;
}

export interface TripResponse {
  result: string;
}
