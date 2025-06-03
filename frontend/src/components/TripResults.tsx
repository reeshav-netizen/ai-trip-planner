import React from 'react';
import {
  Box,
  Typography,
  Button,
  Divider,
  Paper
} from '@mui/material';
import { Refresh } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { TripResponse } from '../types/trip';

interface TripResultsProps {
  response: TripResponse;
  onNewTrip: () => void;
}

const TripResults: React.FC<TripResultsProps> = ({ response, onNewTrip }) => {
  return (
    <Box>
      {/* Results Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 2 }}>
        <Typography variant="h6" sx={{ fontSize: '2rem' }}>
          🧳
        </Typography>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Your Comprehensive Trip Plan
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Results Content */}
      <Paper
        sx={{
          p: 3,
          bgcolor: 'grey.50',
          maxHeight: '600px',
          overflow: 'auto',
          '& h1, & h2, & h3': {
            color: 'primary.main',
            marginTop: 2,
            marginBottom: 1
          },
          '& h1': { fontSize: '1.5rem' },
          '& h2': { fontSize: '1.3rem' },
          '& h3': { fontSize: '1.1rem' },
          '& p': {
            marginBottom: 1.5,
            lineHeight: 1.7
          },
          '& ul, & ol': {
            paddingLeft: 3,
            marginBottom: 2
          },
          '& li': {
            marginBottom: 0.5
          },
          '& strong': {
            fontWeight: 600,
            color: 'text.primary'
          },
          '& em': {
            fontStyle: 'italic',
            color: 'text.secondary'
          }
        }}
      >
        <ReactMarkdown>
          {response.result}
        </ReactMarkdown>
      </Paper>

      {/* Action Button */}
      <Box sx={{ mt: 3, textAlign: 'center' }}>
        <Button
          variant="outlined"
          onClick={onNewTrip}
          startIcon={<Refresh />}
          sx={{ px: 4 }}
        >
          Plan Another Trip
        </Button>
      </Box>
    </Box>
  );
};

export default TripResults;
