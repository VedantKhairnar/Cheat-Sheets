import { HashRouter as Router, Route, Switch } from 'react-router-dom';
import Landing from './pages/Landing';
import Header from './components/Header';
import Topics from './pages/Topics';
import AboutUs from './pages/AboutUs';
import NotFound from './pages/NotFound';

function App() {

  return (
    <Router>
      <Header />

      <Switch>
        <Route path="/" exact>
          <Landing />
        </Route>
        <Route path="/topics" exact>
          <Topics />
        </Route>
        <Route path="/aboutus" exact>
          <AboutUs />
        </Route>
        <Route>
          <NotFound />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
