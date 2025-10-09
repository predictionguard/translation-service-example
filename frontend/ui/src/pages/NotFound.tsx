import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Box, Typography, Button, Container } from "@mui/material";
import { Link } from "react-router-dom";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          textAlign: 'center',
        }}
      >
        <Typography variant="h1" component="h1" sx={{ mb: 2, fontSize: '4rem', fontWeight: 'bold' }}>
          404
        </Typography>
        <Typography variant="h5" component="p" sx={{ mb: 3, color: 'text.secondary' }}>
          Oops! Page not found
        </Typography>
        <Button
          component={Link}
          to="/"
          variant="contained"
          color="primary"
          sx={{ textDecoration: 'none' }}
        >
          Return to Home
        </Button>
      </Box>
    </Container>
  );
};

export default NotFound;

