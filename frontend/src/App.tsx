import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Paper,
  Card,
  CardContent,
  Chip
} from '@mui/material';
import { Flight, TravelExplore } from '@mui/icons-material';
import TripPlannerForm from './components/TripPlannerForm';
import TripResults from './components/TripResults';
import { TripRequest, TripResponse } from './types/trip';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h3: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
});

function App() {
  const [tripResponse, setTripResponse] = useState<TripResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePlanTrip = async (tripRequest: TripRequest) => {
    setLoading(true);
    setError(null);
    setTripResponse(null);

    try {
      const response = await fetch('http://localhost:8000/plan-trip', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(tripRequest),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: TripResponse = await response.json();
      setTripResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error planning trip:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNewTrip = () => {
    setTripResponse(null);
    setError(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <TravelExplore sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              AI Trip Planner
            </Typography>
            <Flight />
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          {/* Hero Section */}
          <Paper
            sx={{
              p: 4,
              mb: 4,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white'
            }}
          >
            <Typography variant="h3" component="h1" gutterBottom align="center">
              Plan Your Perfect Trip
            </Typography>
            <Typography variant="h6" align="center" sx={{ opacity: 0.9 }}>
              Let our AI agents help you discover amazing destinations, create itineraries,
              manage budgets, and find local experiences
            </Typography>
          </Paper>

          {/* Features Cards */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 4 }}>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🔍 Research
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Discover destinations, weather, attractions, and local culture
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📅 Itineraries
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Get detailed day-by-day travel plans and schedules
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    💰 Budget
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Smart budget planning and money-saving tips
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🍽️ Local
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Authentic experiences and hidden gems
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </Box>

          {/* Main Content */}
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
            <Box sx={{ flex: '0 0 auto', width: { xs: '100%', md: '400px' } }}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Plan Your Trip
                </Typography>
                <TripPlannerForm onSubmit={handlePlanTrip} loading={loading} />
              </Paper>
            </Box>

            <Box sx={{ flex: 1 }}>
              {error && (
                <Paper sx={{ p: 3, mb: 2, bgcolor: 'error.light', color: 'error.contrastText' }}>
                  <Typography variant="h6" gutterBottom>
                    Error
                  </Typography>
                  <Typography>{error}</Typography>
                </Paper>
              )}

              {tripResponse && (
                <Paper sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h5" gutterBottom>
                      Your Trip Plan
                    </Typography>
                    <Chip
                      label={tripResponse.agent_type}
                      color="primary"
                      variant="outlined"
                    />
                  </Box>
                  <TripResults response={tripResponse} onNewTrip={handleNewTrip} />
                </Paper>
              )}

              {!tripResponse && !loading && !error && (
                <Paper sx={{ p: 6, textAlign: 'center', bgcolor: 'grey.50' }}>
                  <Typography variant="h6" color="text.secondary">
                    Fill out the form to get your personalized trip plan
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Our AI agents will analyze your preferences and create the perfect itinerary
                  </Typography>
                </Paper>
              )}
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
